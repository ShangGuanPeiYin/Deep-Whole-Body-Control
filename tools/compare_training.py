"""Fail-closed three-seed comparison for Gate F."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def median_within_interval(values: np.ndarray, lower: float, upper: float) -> bool:
    median = float(np.median(np.asarray(values, dtype=np.float64)))
    return lower <= median <= upper


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"missing metrics file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty metrics file: {path}")
    return rows


def compare_training_runs(config: dict, base_dir: Path | None = None) -> dict:
    required = set(config.get("required_seeds", []))
    supplied = {int(run["seed"]) for run in config.get("runs", [])}
    missing = sorted(required - supplied)
    unexpected = sorted(supplied - required)
    if missing:
        raise ValueError(f"missing required seeds: {missing}")
    if unexpected:
        raise ValueError(f"unexpected seeds: {unexpected}")
    if required != {1, 2, 3}:
        raise ValueError("Gate F requires exactly seeds 1, 2 and 3")
    if config.get('baseline_status') == 'pending':
        raise ValueError('Gate F baseline intervals are pending real legacy three-seed training')
    if not config.get('gates'):
        raise ValueError('Gate F requires nonempty predeclared metric intervals')
    if len(config['runs']) != 3:
        raise ValueError('Gate F requires exactly one run for each seed')
    base_dir = base_dir or Path.cwd()
    series = {}
    for run in config["runs"]:
        path = Path(run["metrics"])
        if not path.is_absolute():
            path = base_dir / path
        series[int(run["seed"])] = _read_jsonl(path)
    results = {}
    gate_passed = True
    for metric, gate in config.get("gates", {}).items():
        final_values = []
        threshold_iterations = []
        threshold = float(gate["threshold"])
        for seed in sorted(required):
            rows = series[seed]
            if any(metric not in row or "iteration" not in row for row in rows):
                raise ValueError(f"seed {seed} is missing metric {metric!r} or iteration")
            final_values.append(float(rows[-1][metric]))
            hits = [int(row["iteration"]) for row in rows if float(row[metric]) >= threshold]
            if not hits:
                gate_passed = False
                threshold_iterations.append(None)
            else:
                threshold_iterations.append(min(hits))
        values = np.asarray(final_values)
        if not np.isfinite(values).all():
            raise ValueError(f'non-finite metric: {metric}')
        lower, upper = map(float, gate["final_median"])
        interval_passed = median_within_interval(values, lower, upper)
        gate_passed &= interval_passed
        finite_hits = [value for value in threshold_iterations if value is not None]
        results[metric] = {
            "values": final_values,
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "interval": [lower, upper],
            "interval_passed": interval_passed,
            "threshold": threshold,
            "threshold_iterations": threshold_iterations,
            # First iteration by which every seed that crossed has crossed.
            "first_threshold_iteration": max(finite_hits) if len(finite_hits) == len(required) else None,
        }
    return {"gate_f_passed": bool(gate_passed), "required_seeds": sorted(required), "metrics": results}


def _markdown(report: dict, config_hash: str) -> str:
    status = "PASS" if report["gate_f_passed"] else "FAIL"
    lines = ["# Gate F Training Comparison", "", f"Status: **{status}**", "", f"Config SHA-256: `{config_hash}`", ""]
    for name, result in report["metrics"].items():
        lines.extend([
            f"## {name}", "",
            f"Final seed values: `{result['values']}`", "",
            f"Mean / median / std: `{result['mean']:.6g}` / `{result['median']:.6g}` / `{result['std']:.6g}`", "",
            f"Accepted median interval: `{result['interval']}`; passed: `{result['interval_passed']}`", "",
            f"Threshold iterations: `{result['threshold_iterations']}`", "",
        ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    report = compare_training_runs(config, Path.cwd())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(raw).hexdigest()
    if args.out.suffix == ".md":
        args.out.write_text(_markdown(report, config_hash), encoding="utf-8")
    else:
        args.out.write_text(json.dumps({**report, "config_sha256": config_hash}, indent=2) + "\n", encoding="utf-8")
    return 0 if report["gate_f_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
