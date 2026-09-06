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


def _normalize_colliders(colliders: Iterable[Mapping]) -> dict[str, dict]:
    """Produce a deterministic, shape-level collision contract.

    A body may contain more than one collision shape of the same type.  The
    ordinal is therefore part of the key after sorting the shape descriptors,
    while the descriptor itself remains JSON-only and independent of USD.
    """
    normalized: list[dict] = []
    for collider in colliders:
        normalized.append(
            {
                "body": str(collider["body"]),
                "shape": str(collider["shape"]).lower(),
                "local_position": [float(value) for value in collider.get("local_position", ())],
                "local_rotation": [float(value) for value in collider.get("local_rotation", ())],
                "dimensions": [float(value) for value in collider.get("dimensions", ())],
            }
        )
    normalized.sort(
        key=lambda row: (
            row["body"], row["shape"], row["local_position"], row["local_rotation"], row["dimensions"]
        )
    )
    result: dict[str, dict] = {}
    duplicates: dict[str, int] = {}
    for row in normalized:
        stem = f'{row["body"]}/{row["shape"]}'
        ordinal = duplicates.get(stem, 0)
        duplicates[stem] = ordinal + 1
        result[f"{stem}#{ordinal}"] = {
            field: row[field] for field in ("local_position", "local_rotation", "dimensions")
        }
    return result


def build_asset_report(
    joints: Iterable[Mapping],
    bodies: Iterable[Mapping],
    collider_count: int = 0,
    *,
    colliders: Iterable[Mapping] | None = None,
) -> dict:
    joint_rows = sorted(joints, key=lambda row: row["name"])
    body_rows = sorted(bodies, key=lambda row: row["name"])
    report = {
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
    if colliders is not None:
        report["colliders"] = _normalize_colliders(colliders)
        report["collider_count"] = len(report["colliders"])
    return report


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
    if "collider_geometry" in tolerances:
        geometry_tolerance = tolerances["collider_geometry"]
        reference_colliders = reference.get("colliders")
        candidate_colliders = candidate.get("colliders")
        if not isinstance(reference_colliders, Mapping) or not isinstance(candidate_colliders, Mapping):
            failures.append("missing required collider geometry")
        else:
            reference_keys = set(reference_colliders)
            candidate_keys = set(candidate_colliders)
            failures.extend(f"missing collider: {name}" for name in sorted(reference_keys - candidate_keys))
            failures.extend(f"unexpected collider: {name}" for name in sorted(candidate_keys - reference_keys))
            for name in sorted(reference_keys & candidate_keys):
                display_name = name.rsplit("#", 1)[0]
                for field in ("local_position", "local_rotation", "dimensions"):
                    expected = np.asarray(reference_colliders[name].get(field, ()), dtype=float)
                    actual = np.asarray(candidate_colliders[name].get(field, ()), dtype=float)
                    if expected.shape != actual.shape or not np.allclose(
                        actual,
                        expected,
                        atol=float(geometry_tolerance["atol"]),
                        rtol=float(geometry_tolerance["rtol"]),
                    ):
                        failures.append(f"collider {field} drift for {display_name}")
    for field, tolerance in tolerances.items():
        if field == "collider_geometry":
            continue
        reference_values = reference.get(field, {})
        candidate_values = candidate.get(field, {})
        body_field = field in {"mass", "inertia", "inertia_tensor", "center_of_mass"}
        for side, values, names in (
            ("reference", reference_values, reference_bodies if body_field else reference_names),
            ("candidate", candidate_values, candidate_bodies if body_field else candidate_names),
        ):
            for name in sorted(names - set(values)):
                failures.append(f'{side} {field}: missing declared entity property for {name}')
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
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path.resolve()))
    if stage is None:
        raise ValueError(f"could not open USD stage: {path}")
    joints = []
    bodies = []
    colliders = []
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
            matrix = UsdGeom.Xformable(prim).GetLocalTransformation()
            translation = matrix.ExtractTranslation()
            rotation = matrix.ExtractRotationQuat()
            shape_prim = prim
            if not any(
                prim.IsA(schema) for schema in (UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube, UsdGeom.Mesh)
            ):
                for descendant in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                    if any(
                        descendant.IsA(schema)
                        for schema in (UsdGeom.Sphere, UsdGeom.Capsule, UsdGeom.Cube, UsdGeom.Mesh)
                    ):
                        shape_prim = descendant
                        break
            shape = shape_prim.GetTypeName().lower()
            dimensions: list[float] = []
            if shape_prim.IsA(UsdGeom.Sphere):
                dimensions = [float(UsdGeom.Sphere(shape_prim).GetRadiusAttr().Get())]
            elif shape_prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(shape_prim)
                dimensions = [float(capsule.GetRadiusAttr().Get()), float(capsule.GetHeightAttr().Get())]
            elif shape_prim.IsA(UsdGeom.Cube):
                dimensions = [float(UsdGeom.Cube(shape_prim).GetSizeAttr().Get())]
            elif shape_prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(shape_prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
                face_indices = mesh.GetFaceVertexIndicesAttr().Get()
                dimensions = [
                    float(len(points)),
                    float(len(face_indices)),
                    *np.min(points, axis=0).tolist(),
                    *np.max(points, axis=0).tolist(),
                ]
            collision_container = prim
            while collision_container.GetName() != "collisions":
                collision_container = collision_container.GetParent()
            body = canonicalize_body_name(collision_container.GetParent().GetName())
            colliders.append(
                {
                    "body": body,
                    "shape": shape,
                    "local_position": [float(value) for value in translation],
                    "local_rotation": [
                        float(rotation.GetReal()),
                        *[float(value) for value in rotation.GetImaginary()],
                    ],
                    "dimensions": dimensions,
                }
            )
    return build_asset_report(joints, bodies, colliders=colliders)


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
        sim = None
        try:
            # Isaac Sim 5.1 may shut down an app that only opens a detached USD
            # stage.  Retain a minimal simulation context while auditing.
            import isaaclab.sim as sim_utils
            sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args.device))
            report = inspect_usd(args.usd)
            args.out.write_text(json.dumps(report, indent=2) + "\n")
        except BaseException:
            # Kit occasionally suppresses Python's default exception hook while
            # shutting down.  Preserve the actual diagnostic for the gate log.
            import traceback
            traceback.print_exc()
            raise
        finally:
            if sim is not None:
                type(sim).clear_instance()
            launcher.app.close()
        return 0
    if not (args.reference and args.candidate and args.tolerances):
        parser.error("provide --usd, or all of --reference --candidate --tolerances")
    result = compare_asset_report(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()), json.loads(args.tolerances.read_text()))
    args.out.write_text(json.dumps({"passed": result.passed, "failures": result.failures}, indent=2) + "\n")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
