import torch


def test_osc_uses_mass_weighted_pose_feedback_and_gravity():
    from dwbc_isaaclab.tasks.widow_go1.control import compute_osc_torques

    eye = torch.eye(6).unsqueeze(0)
    body_jacobians = eye[:, None].repeat(1, 2, 1, 1)
    actual = compute_osc_torques(
        2 * eye, eye, torch.full((1, 6), 0.1), torch.ones(1, 6),
        torch.full((6,), 100.), torch.full((6,), 20.),
        body_jacobians, torch.tensor([[1., 2.]]),
    )
    torch.testing.assert_close(actual, torch.tensor([[-20., -20., 9.43, -20., -20., -20.]]))


def test_singular_osc_jacobian_retains_gravity_compensation():
    from dwbc_isaaclab.tasks.widow_go1.control import compute_osc_torques

    body_jacobians = torch.eye(6)[None, None]
    actual = compute_osc_torques(
        torch.eye(6)[None], torch.zeros(1, 6, 6), torch.ones(1, 6),
        torch.zeros(1, 6), torch.ones(6), torch.ones(6),
        body_jacobians, torch.tensor([[2.]]),
    )
    torch.testing.assert_close(actual, torch.tensor([[0., 0., 19.62, 0., 0., 0.]]))


def test_osc_matches_original_legacy_method():
    import ast
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import pytest
    from dwbc_isaaclab.tasks.widow_go1.control import compute_osc_torques

    root = Path(os.environ.get('DWBC_LEGACY_ROOT', '/home/xxs/research/Deep-Whole-Body-Control'))
    task = root / 'legged_gym/legged_gym/envs/widowGo1/widowGo1.py'
    gym_math = root.parent / 'isaacgym/python/isaacgym/torch_utils.py'
    if not task.exists() or not gym_math.exists():
        pytest.skip('legacy OSC oracle unavailable')
    namespace = {'torch': torch}
    for path, names in (
        (gym_math, {'quat_mul', 'quat_conjugate', 'quat_apply'}),
        (root / 'legged_gym/legged_gym/utils/math.py', {'orientation_error'}),
        (task, {'get_arm_ee_control_torques'}),
    ):
        definitions = [node for node in ast.walk(ast.parse(path.read_text()))
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in definitions:
            # Drop only JIT decorators: source bodies are the unchanged oracle.
            node.decorator_list = []
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    torch.manual_seed(19)
    n = 3
    eye = torch.eye(6, dtype=torch.float64).expand(n, -1, -1)
    jacobian = eye + 0.05 * torch.randn(n, 6, 6, dtype=torch.float64)
    whole = torch.randn(n, 9, 6, 8, dtype=torch.float64)
    masses = torch.rand(n, 9, dtype=torch.float64)
    gravity = torch.zeros(n, 9, 6, 1, dtype=torch.float64)
    gravity[:, :, 2, 0] = 9.81 * masses
    identity = torch.tensor([[0., 0., 0., 1.]], dtype=torch.float64).repeat(n, 1)
    env = SimpleNamespace(
        mm=2 * eye, ee_j_eef=jacobian, jacobian_whole=whole, g_force=gravity,
        ee_orn=identity, ee_orn_des=identity, base_yaw_quat=identity,
        root_states=torch.zeros(n, 13, dtype=torch.float64),
        z_invariant_offset=torch.full((n, 1), .53, dtype=torch.float64),
        curr_ee_goal_cart=torch.randn(n, 3, dtype=torch.float64),
        ee_pos=torch.randn(n, 3, dtype=torch.float64),
        ee_vel=torch.randn(n, 6, dtype=torch.float64),
        arm_osc_kp=torch.tensor([100., 100., 100., 30., 30., 30.], dtype=torch.float64),
    )
    env.arm_osc_kd = 2 * env.arm_osc_kp.sqrt()
    pose_error = torch.cat((env.curr_ee_goal_cart - env.ee_pos, torch.zeros(n, 3, dtype=torch.float64)), -1)
    pose_error[:, 2] += .53
    expected = namespace['get_arm_ee_control_torques'](env)
    actual = compute_osc_torques(env.mm, jacobian, pose_error, env.ee_vel,
                                env.arm_osc_kp, env.arm_osc_kd, whole[..., :6], masses)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)
