"""Convert the frozen WidowGo1 URDF with Isaac Lab 2.3.2."""

import argparse
import json
import numpy as np
from pathlib import Path


def importer_settings() -> dict[str, object]:
    return {
        "fix_base": False,
        "merge_fixed_joints": True,
        "force_usd_conversion": True,
        "joint_drive_target_type": "none",
        "joint_stiffness": 0.0,
        "joint_damping": 0.0,
        "replace_cylinders_with_capsules": True,
    }


def default_asset_paths() -> tuple[Path, Path]:
    asset_root = Path(__file__).resolve().parents[1]
    return asset_root / "source_urdf/urdf/widowGo1.urdf", asset_root / "usd/widow_go1.usd"


def canonical_body_name(usd_name: str) -> str:
    if usd_name.startswith("wx250s_"):
        return "wx250s/" + usd_name[len("wx250s_"):]
    return usd_name


def apply_legacy_mass_properties(usd_path: Path, contract_path: Path) -> None:
    """Author the exact legacy runtime mass properties into the converter's physics layer."""
    from pxr import Gf, Usd, UsdPhysics

    contract = json.loads(contract_path.read_text())
    physics_path = usd_path.parent / "configuration" / f"{usd_path.stem}_physics.usd"
    stage = Usd.Stage.Open(str(physics_path))
    if stage is None:
        raise RuntimeError(f"could not open generated physics layer: {physics_path}")
    matched: set[str] = set()
    for prim in stage.TraverseAll():
        body_name = canonical_body_name(prim.GetName())
        if body_name not in contract["mass"]:
            continue
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.GetMassAttr().Set(float(contract["mass"][body_name]))
        if 'inertia_tensor' not in contract or 'center_of_mass' not in contract:
            raise ValueError('full legacy inertia tensor and center of mass are required')
        tensor = np.asarray(contract['inertia_tensor'][body_name], dtype=float)
        tensor = (tensor + tensor.T) / 2
        moments, axes = np.linalg.eigh(tensor)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        if np.any(moments <= 0):
            raise ValueError(f'nonpositive inertia for {body_name}')
        # Gf matrices use row-vector rotations; eigenvectors are columns in NumPy.
        quaternion = Gf.Matrix3d(*axes.T.flatten().tolist()).ExtractRotation().GetQuat()
        mass_api.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*moments.tolist()))
        mass_api.GetPrincipalAxesAttr().Set(Gf.Quatf(quaternion))
        mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*contract['center_of_mass'][body_name]))
        matched.add(body_name)
    missing = sorted(set(contract["mass"]) - matched)
    if missing:
        raise RuntimeError(f"mass-property contract bodies absent from generated USD: {missing}")
    stage.GetRootLayer().Save()


def main() -> int:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    default_input, default_output = default_asset_paths()
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument('--mass-contract',type=Path,default=default_output.parents[1] / 'legacy_full_mass_properties.json')
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    try:
        from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

        settings = importer_settings()
        cfg = UrdfConverterCfg(
            asset_path=str(args.input.resolve()),
            usd_dir=str(args.output.resolve().parent),
            usd_file_name=args.output.name,
            fix_base=settings["fix_base"],
            merge_fixed_joints=settings["merge_fixed_joints"],
            force_usd_conversion=settings["force_usd_conversion"],
            replace_cylinders_with_capsules=settings["replace_cylinders_with_capsules"],
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=settings["joint_stiffness"], damping=settings["joint_damping"]
                ),
                target_type=settings["joint_drive_target_type"],
            ),
        )
        converter = UrdfConverter(cfg)
        if Path(converter.usd_path).resolve() != args.output.resolve():
            raise RuntimeError(f"converter wrote unexpected path: {converter.usd_path}")
        contract_path = args.mass_contract.resolve()
        apply_legacy_mass_properties(args.output.resolve(), contract_path)
        print(converter.usd_path)
        return 0
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
