"""Compare serialized robot asset reports without importing Isaac Sim."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class AssetComparison:
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def compare_asset_report(reference: Mapping, candidate: Mapping, tolerances: Mapping) -> AssetComparison:
    failures: list[str] = []
    reference_names = set(reference.get("joint_names", ()))
    candidate_names = set(candidate.get("joint_names", ()))
    failures.extend(f"missing joint: {name}" for name in sorted(reference_names - candidate_names))
    failures.extend(f"unexpected joint: {name}" for name in sorted(candidate_names - reference_names))
    for field, tolerance in tolerances.items():
        reference_values = reference.get(field, {})
        candidate_values = candidate.get(field, {})
        for name in sorted(set(reference_values) & set(candidate_values)):
            expected = np.asarray(reference_values[name], dtype=float)
            actual = np.asarray(candidate_values[name], dtype=float)
            if expected.shape != actual.shape or not np.allclose(
                actual, expected, atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"])
            ):
                max_error = float(np.max(np.abs(actual - expected))) if expected.shape == actual.shape else float("inf")
                failures.append(f"{field} drift for {name}: max_abs_error={max_error}")
    return AssetComparison(tuple(failures))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tolerances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare_asset_report(
        json.loads(args.reference.read_text()),
        json.loads(args.candidate.read_text()),
        json.loads(args.tolerances.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"passed": result.passed, "failures": result.failures}, indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
