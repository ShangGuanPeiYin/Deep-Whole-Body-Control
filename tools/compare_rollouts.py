"""Fail-closed quantitative comparison of legacy and Isaac Lab traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FieldComparison:
    name: str
    passed: bool
    shape: tuple[int, ...]
    max_abs_error: float
    rms_error: float
    max_relative_error: float
    first_bad_step: int | None


def compare_field(
    name: str, reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float
) -> FieldComparison:
    if reference.shape != candidate.shape:
        return FieldComparison(name, False, tuple(candidate.shape), float("inf"), float("inf"), float("inf"), 0)
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.size == 0 or not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        return FieldComparison(name, False, tuple(candidate.shape), float('inf'), float('inf'), float('inf'), 0)
    difference = np.abs(candidate.astype(float) - reference.astype(float))
    close = np.isclose(candidate, reference, atol=atol, rtol=rtol, equal_nan=False)
    bad = np.argwhere(~close)
    first_bad_step = int(bad[0, 0]) if bad.size else None
    denominator = np.maximum(np.abs(reference.astype(float)), 1.0e-12)
    return FieldComparison(
        name=name,
        passed=bool(np.all(close)),
        shape=tuple(candidate.shape),
        max_abs_error=float(np.max(difference)) if difference.size else 0.0,
        rms_error=float(np.sqrt(np.mean(difference**2))) if difference.size else 0.0,
        max_relative_error=float(np.max(difference / denominator)) if difference.size else 0.0,
        first_bad_step=first_bad_step,
    )


def compare_metadata(reference: Mapping, candidate: Mapping) -> tuple[str, ...]:
    keys = (
        "schema_version", "task", "seed", "steps", "num_envs", "config_sha256",
        "action_sha256", "action_order", "joint_order", "tensor_shapes",
        "initial_snapshot_sha256",
    )
    failures = [f"missing required metadata: {key}" for key in keys
                if key != 'initial_snapshot_sha256' and (key not in reference or key not in candidate)]
    failures.extend(f"metadata mismatch: {key}" for key in keys if reference.get(key) != candidate.get(key))
    return tuple(failures)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    reference_metadata = json.loads((args.legacy.parent / "metadata.json").read_text())
    candidate_metadata = json.loads((args.candidate.parent / "metadata.json").read_text())
    metadata_failures = compare_metadata(reference_metadata, candidate_metadata)
    tolerances = json.loads(args.tolerances.read_text())
    reference = np.load(args.legacy)
    candidate = np.load(args.candidate)
    fields = {}
    schema_failures = list(metadata_failures)
    expected_fields = set(tolerances["fields"])
    if set(reference.files) != set(candidate.files):
        schema_failures.append("trace field set mismatch")
    if set(reference.files) != expected_fields:
        schema_failures.append("tolerance field set mismatch")
    for name, limits in tolerances["fields"].items():
        if name not in reference or name not in candidate:
            continue
        fields[name] = compare_field(
            name, reference[name], candidate[name], float(limits["atol"]), float(limits["rtol"])
        )
    passed = not schema_failures and all(result.passed for result in fields.values())
    report = {
        "gate_b_to_d_passed": passed,
        "schema_failures": schema_failures,
        "fields": {name: asdict(result) for name, result in fields.items()},
        "inputs": {
            "legacy": str(args.legacy),
            "candidate": str(args.candidate),
            "tolerances": str(args.tolerances),
            "tolerances_sha256": _sha256(args.tolerances),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
