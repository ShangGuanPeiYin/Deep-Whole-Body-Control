import numpy as np

from tools.compare_rollouts import compare_field, compare_metadata


def test_abs_and_relative_tolerance_are_reported():
    result = compare_field("dof_pos", np.array([[1.0]]), np.array([[1.001]]), 0.01, 0.01)
    assert result.passed
    assert result.max_abs_error == 0.0009999999999998899
    assert result.first_bad_step is None


def test_first_bad_step_uses_leading_time_dimension():
    result = compare_field(
        "obs", np.zeros((3, 2, 1)), np.array([[[0.0], [0.0]], [[0.0], [2.0]], [[0.0], [0.0]]]), 0.1, 0.0
    )
    assert not result.passed
    assert result.first_bad_step == 1


def test_metadata_comparison_fails_closed_on_action_order():
    reference = {"schema_version": 1, "action_order": ["a", "b"], "seed": 1}
    candidate = {"schema_version": 1, "action_order": ["b", "a"], "seed": 1}
    assert "metadata mismatch: action_order" in compare_metadata(reference, candidate)


def test_identical_infinities_and_empty_traces_cannot_pass_alignment():
    for array in (np.array([[np.inf]]), np.empty((0, 2))):
        assert not compare_field('state', array, array, 0.0, 0.0).passed


def test_missing_metadata_on_both_sides_cannot_pass_alignment():
    assert 'missing required metadata: action_order' in compare_metadata({}, {})
