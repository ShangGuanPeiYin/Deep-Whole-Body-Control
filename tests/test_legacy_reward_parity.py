"""Same-state oracle: execute all active original reward functions unchanged."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dwbc_isaaclab.tasks.widow_go1.rewards import leg_reward, arm_reward


def test_every_active_reward_term_matches_legacy_on_shared_states():
    root = Path(os.environ.get('DWBC_LEGACY_ROOT', '/home/xxs/research/Deep-Whole-Body-Control'))
    task = root / 'legged_gym/legged_gym/envs/widowGo1/widowGo1.py'
    gym_math = root.parent / 'isaacgym/python/isaacgym/torch_utils.py'
    if not task.exists() or not gym_math.exists():
        pytest.skip('legacy reward oracle unavailable')
    leg_scales = dict(survive=.2, tracking_lin_vel_x_l1=.5, tracking_ang_vel_yaw_exp=.15,
                      hip_action_l2=-.01, foot_contacts_z=-1e-4, energy_square=-6e-5)
    arm_scales = dict(tracking_ee_sphere=.55, arm_energy_abs_sum=-.004)
    namespace = {'torch': torch}
    for path, names in (
        (gym_math, {'quat_rotate_inverse', 'quat_apply'}),
        (root / 'legged_gym/legged_gym/utils/math.py', {'cart2sphere'}),
        (task, {'_reward_' + name for name in (*leg_scales, *arm_scales)}),
    ):
        definitions = [node for node in ast.walk(ast.parse(path.read_text()))
                       if isinstance(node, ast.FunctionDef) and node.name in names]
        for node in definitions:
            node.decorator_list = []
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    torch.manual_seed(37)
    n = 9
    actions, torques, velocities = torch.randn(n, 18), torch.randn(n, 20), torch.randn(n, 20)
    motor_order = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17]
    joint_order = motor_order + [18, 19]
    foot_order = [1, 0, 3, 2]
    wrench = torch.randn(n, 4, 6) * 20
    ee_local = torch.randn(n, 3)
    yaw = torch.linspace(-2.9, 2.9, n)
    yaw_quat = torch.stack((torch.zeros(n), torch.zeros(n), (yaw / 2).sin(), (yaw / 2).cos()), -1)
    root_state = torch.randn(n, 13)
    reference = torch.cat((root_state[:, :2], torch.full((n, 1), .53)), -1)
    env = SimpleNamespace(
        num_envs=n, device='cpu', actions=actions[:, motor_order], torques=torques[:, joint_order],
        dof_vel=velocities[:, joint_order], base_lin_vel=torch.randn(n, 3), base_ang_vel=torch.randn(n, 3),
        commands=torch.randn(n, 3), force_sensor_tensor=wrench[:, foot_order],
        base_yaw_quat=yaw_quat, root_states=root_state, z_invariant_offset=torch.full((n, 1), .53),
        ee_pos=reference + namespace['quat_apply'](yaw_quat, ee_local), curr_ee_goal_sphere=torch.randn(n, 3),
        sphere_error_scale=torch.tensor([2., 5 / (3 * torch.pi), 5 / (6 * torch.pi)]),
        cfg=SimpleNamespace(rewards=SimpleNamespace(tracking_sigma=1., tracking_ee_sigma=1.)),
        episode_metric_sums={name: torch.zeros(n) for name in (
            'tracking_lin_vel_x_l1', 'tracking_ang_vel_yaw_exp', 'leg_action_l2',
            'foot_contacts_z', 'energy_square', 'tracking_ee_sphere',
        )},
    )
    leg, leg_terms = leg_reward(actions=actions, torques=torques, joint_vel=velocities,
                               base_lin_vel=env.base_lin_vel, base_ang_vel=env.base_ang_vel,
                               commands=env.commands, foot_force_z=wrench[:, :, 2])
    arm, arm_terms = arm_reward(ee_position_local=ee_local, ee_goal_sphere=env.curr_ee_goal_sphere,
                               torques=torques, joint_vel=velocities)
    for scales, terms, reward in ((leg_scales, leg_terms, leg), (arm_scales, arm_terms, arm)):
        expected_sum = torch.zeros(n)
        assert set(terms) == set(scales)
        for name, scale in scales.items():
            expected = namespace['_reward_' + name](env) * scale
            torch.testing.assert_close(terms[name], expected, atol=1e-7, rtol=1e-6)
            expected_sum += expected
        torch.testing.assert_close(reward, expected_sum / 100, atol=1e-7, rtol=1e-6)
