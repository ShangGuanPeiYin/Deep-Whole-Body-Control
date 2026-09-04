from tools.audit_usd import compare_asset_report


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
