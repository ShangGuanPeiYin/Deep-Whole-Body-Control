from tools.audit_usd import build_asset_report, compare_asset_report, normalize_joint_limits
from assets.widow_go1.import_config.widow_go1_urdf_import import (
    canonical_body_name,
    default_asset_paths,
    importer_settings,
)


def test_asset_report_detects_missing_names_and_numeric_drift():
    reference = {
        "joint_names": ["FR_hip_joint", "widow_elbow"],
        "joint_limits": {"FR_hip_joint": [-1.0, 1.0], "widow_elbow": [-2.0, 2.0]},
    }
    candidate = {
        "joint_names": ["FR_hip_joint", "extra_joint"],
        "joint_limits": {"FR_hip_joint": [-1.0, 1.1], "extra_joint": [-2.0, 2.0]},
    }
    result = compare_asset_report(reference, candidate, {"joint_limits": {"atol": 0.01, "rtol": 0.0}})
    assert not result.passed
    assert any("missing joint: widow_elbow" in failure for failure in result.failures)
    assert any("unexpected joint: extra_joint" in failure for failure in result.failures)
    assert any("FR_hip_joint" in failure and "joint_limits" in failure for failure in result.failures)


def test_asset_report_detects_missing_body_names():
    result = compare_asset_report(
        {"joint_names": [], "body_names": ["trunk"]},
        {"joint_names": [], "body_names": ["base"]},
        {},
    )
    assert "missing body: trunk" in result.failures
    assert "unexpected body: base" in result.failures


def test_asset_report_detects_collider_count_change():
    result = compare_asset_report(
        {"joint_names": [], "body_names": [], "collider_count": 25},
        {"joint_names": [], "body_names": [], "collider_count": 24},
        {},
    )
    assert "collider_count: expected 25, got 24" in result.failures


def test_unbounded_usd_joint_matches_legacy_zero_limit_sentinel():
    assert normalize_joint_limits(float("-inf"), float("inf"), angular=True) == [0.0, 0.0]


def test_importer_preserves_fixed_joint_markers_and_urdf_inertia():
    settings = importer_settings()
    assert settings == {
        "fix_base": False,
        "merge_fixed_joints": True,
        "force_usd_conversion": True,
        "joint_drive_target_type": "none",
        "joint_stiffness": 0.0,
        "joint_damping": 0.0,
    }


def test_default_import_paths_stay_inside_widow_go1_asset():
    source, output = default_asset_paths()
    assert source.as_posix().endswith("assets/widow_go1/source_urdf/urdf/widowGo1.urdf")
    assert output.as_posix().endswith("assets/widow_go1/usd/widow_go1.usd")


def test_importer_reverses_usd_name_sanitization_for_contract_lookup():
    assert canonical_body_name("wx250s_upper_arm_link") == "wx250s/upper_arm_link"
    assert canonical_body_name("FR_thigh") == "FR_thigh"


def test_asset_report_orders_names_and_keeps_joint_properties():
    report = build_asset_report(
        [{"name": "widow_elbow", "limits": [-1.0, 1.0]}, {"name": "FR_hip_joint", "limits": [-0.5, 0.5]}],
        [{"name": "trunk", "mass": 5.0}],
    )
    assert report["joint_names"] == ["FR_hip_joint", "widow_elbow"]
    assert report["joint_limits"]["widow_elbow"] == [-1.0, 1.0]
    assert report["body_names"] == ["trunk"]
