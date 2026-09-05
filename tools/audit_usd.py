"""Compare serialized robot asset reports without importing Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "source"))

from dwbc_isaaclab.tasks.widow_go1.contracts import canonicalize_body_name


@dataclass(frozen=True)
class AssetComparison:
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def build_asset_report(joints: Iterable[Mapping], bodies: Iterable[Mapping], collider_count: int = 0) -> dict:
    joint_rows = sorted(joints, key=lambda row: row["name"])
    body_rows = sorted(bodies, key=lambda row: row["name"])
    return {
        "joint_names": [row["name"] for row in joint_rows],
        "body_names": [row["name"] for row in body_rows],
        "joint_limits": {row["name"]: row["limits"] for row in joint_rows},
        "stiffness": {row["name"]: row.get("stiffness", 0.0) for row in joint_rows},
        "damping": {row["name"]: row.get("damping", 0.0) for row in joint_rows},
        "effort_limit": {row["name"]: row.get("effort_limit", 0.0) for row in joint_rows},
        "mass": {row["name"]: row.get("mass", 0.0) for row in body_rows},
        "inertia": {row["name"]: row.get("inertia", [0.0, 0.0, 0.0]) for row in body_rows},
        "inertia_tensor": {row['name']:row['inertia_tensor'] for row in body_rows if 'inertia_tensor' in row},
        "center_of_mass": {row['name']:row['center_of_mass'] for row in body_rows if 'center_of_mass' in row},
        "collider_count": collider_count,
    }


def normalize_joint_limits(lower: float, upper: float, *, angular: bool) -> list[float]:
    """Normalize USD limits to the legacy Isaac Gym report convention."""
    limits = np.asarray([lower, upper], dtype=float)
    if not np.all(np.isfinite(limits)):
        return [0.0, 0.0]
    if angular:
        limits = np.deg2rad(limits)
    return [float(value) for value in limits]


def compare_asset_report(reference: Mapping, candidate: Mapping, tolerances: Mapping) -> AssetComparison:
    failures: list[str] = []
    reference_names = set(reference.get("joint_names", ()))
    candidate_names = set(candidate.get("joint_names", ()))
    failures.extend(f"missing joint: {name}" for name in sorted(reference_names - candidate_names))
    failures.extend(f"unexpected joint: {name}" for name in sorted(candidate_names - reference_names))
    reference_bodies = set(reference.get("body_names", ()))
    candidate_bodies = set(candidate.get("body_names", ()))
    failures.extend(f"missing body: {name}" for name in sorted(reference_bodies - candidate_bodies))
    failures.extend(f"unexpected body: {name}" for name in sorted(candidate_bodies - reference_bodies))
    reference_colliders = reference.get("collider_count")
    candidate_colliders = candidate.get("collider_count")
    if reference_colliders != candidate_colliders:
        failures.append(f"collider_count: expected {reference_colliders}, got {candidate_colliders}")
    for field, tolerance in tolerances.items():
        reference_values = reference.get(field, {})
        candidate_values = candidate.get(field, {})
        if not reference_values or not candidate_values:
            failures.append(f'missing required physics field: {field}')
        for name in sorted(set(reference_values) ^ set(candidate_values)):
            failures.append(f'{field}: missing property for {name}')
        for name in sorted(set(reference_values) & set(candidate_values)):
            expected = np.asarray(reference_values[name], dtype=float)
            actual = np.asarray(candidate_values[name], dtype=float)
            if expected.shape != actual.shape or not np.allclose(
                actual, expected, atol=float(tolerance["atol"]), rtol=float(tolerance["rtol"])
            ):
                max_error = float(np.max(np.abs(actual - expected))) if expected.shape == actual.shape else float("inf")
                failures.append(f"{field} drift for {name}: max_abs_error={max_error}")
    return AssetComparison(tuple(failures))


def inspect_usd(path: Path) -> dict:
    from pxr import Gf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise ValueError(f"could not open USD stage: {path}")
    joints = []
    bodies = []
    collider_count = 0
    pending = list(stage.GetPseudoRoot().GetFilteredChildren(Usd.TraverseInstanceProxies()))
    while pending:
        prim = pending.pop()
        pending.extend(prim.GetFilteredChildren(Usd.TraverseInstanceProxies()))
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            limits = normalize_joint_limits(
                joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get(), angular=True
            )
            drive_kind = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint = UsdPhysics.PrismaticJoint(prim)
            limits = normalize_joint_limits(
                joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get(), angular=False
            )
            drive_kind = "linear"
        else:
            joint = None
        if joint is not None:
            drive = UsdPhysics.DriveAPI.Get(prim, drive_kind)
            joints.append({
                "name": prim.GetName(),
                "limits": limits,
                "stiffness": float(drive.GetStiffnessAttr().Get() or 0.0),
                "damping": float(drive.GetDampingAttr().Get() or 0.0),
                "effort_limit": float(drive.GetMaxForceAttr().Get() or 0.0),
            })
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass_api = UsdPhysics.MassAPI.Get(stage, prim.GetPath())
            inertia = mass_api.GetDiagonalInertiaAttr().Get() or (0.0, 0.0, 0.0)
            axes = mass_api.GetPrincipalAxesAttr().Get()
            rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(axes))) if axes else np.eye(3)
            full_inertia = rotation.T @ np.diag(inertia) @ rotation
            bodies.append({
                "name": canonicalize_body_name(prim.GetName()),
                "mass": float(mass_api.GetMassAttr().Get() or 0.0),
                "inertia": np.diag(full_inertia).tolist(),
                "inertia_tensor": full_inertia.tolist(),
                "center_of_mass": list(mass_api.GetCenterOfMassAttr().Get()),
            })
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collider_count += 1
    return build_asset_report(joints, bodies, collider_count)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usd", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--tolerances", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    AppLauncher = None
    if "--usd" in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.usd is not None:
        assert AppLauncher is not None
        launcher = AppLauncher(args)
        try:
            report = inspect_usd(args.usd)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        finally:
            launcher.app.close()
        return 0
    if not (args.reference and args.candidate and args.tolerances):
        parser.error("provide --usd, or all of --reference --candidate --tolerances")
    result = compare_asset_report(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()), json.loads(args.tolerances.read_text()))
    args.out.write_text(json.dumps({"passed": result.passed, "failures": result.failures}, indent=2) + "\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
