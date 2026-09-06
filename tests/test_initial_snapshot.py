import numpy as np
import pytest


def physics_snapshot():
    return dict(body_names=np.array(['base']), masses=np.ones((2, 1)),
                inertias=np.tile(np.eye(3).reshape(1, 1, 9), (2, 1, 1)), coms=np.zeros((2, 1, 3)),
                box_masses=np.ones((2, 1)), box_inertias=np.tile(np.eye(3).reshape(1, 9), (2, 1)),
                box_coms=np.zeros((2, 3)), box_materials=np.ones((2, 1, 3)))


def test_partial_shared_snapshot_cannot_silently_keep_random_box_mass():
    from tools.initial_snapshot import validate_physics_snapshot
    snapshot = physics_snapshot()
    del snapshot['box_masses']
    with pytest.raises(ValueError, match='box_masses'):
        validate_physics_snapshot(snapshot, 2)


def test_complete_snapshot_accepts_distinct_robot_and_box_tensor_layouts():
    from tools.initial_snapshot import validate_physics_snapshot
    validate_physics_snapshot(physics_snapshot(), 2)


def test_box_inertia_shape_is_checked_before_simulator_mutation():
    from tools.initial_snapshot import validate_physics_snapshot
    snapshot = physics_snapshot()
    snapshot['box_inertias'] = np.zeros((2, 1, 9))
    with pytest.raises(ValueError, match='box_inertias'):
        validate_physics_snapshot(snapshot, 2)


def test_state_only_fingers_are_clamped_without_changing_controlled_joints():
    from tools.initial_snapshot import sanitize_state_only_joint_positions

    positions = np.array([[.2, 0., 0.], [-.2, .02, -.02]], dtype=np.float32)
    result = sanitize_state_only_joint_positions(
        positions, ('hip', 'left_finger', 'right_finger'),
        np.array([-.5, .015, -.037]), np.array([.5, .037, -.015]),
        ('left_finger', 'right_finger'),
    )
    np.testing.assert_array_equal(result[:, 0], positions[:, 0])
    np.testing.assert_allclose(result[:, 1:], [[.015, -.015], [.02, -.02]])
