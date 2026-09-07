"""Public simulator state plus project-owned state for training continuation.

PhysX internal solver caches are not exposed by this API; restoring this state
does not promise bitwise contact trajectories across separate processes.
"""
import copy
import torch


TENSORS = (
    '_raw_actions', '_actions', '_last_actions', '_torques', '_foot_sensor_wrench',
    '_obs_history', '_commands', '_ee_start_sphere', '_ee_goal_sphere',
    '_current_ee_goal_sphere', '_ee_goal_delta_orientation', '_goal_timer',
    '_trajectory_steps', '_trajectory_total_steps', '_box_offsets', '_env_origins',
    '_mass_params', '_friction_coefficients', '_motor_strength', 'episode_length_buf',
    'reset_buf', 'reset_terminated', 'reset_time_outs',
)
VALUES = ('_curriculum_counter', '_reset_counter', 'common_step_counter',
          '_sim_step_counter', '_lin_range', '_yaw_range', '_goal_ranges')
PROPERTIES = ('masses', 'inertias', 'coms', 'material_properties')


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(clone_tree(item) for item in value)
    return copy.deepcopy(value)


def assert_state_close(actual, expected, path='state'):
    """Validate restored public state before the next simulation step."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.to(expected.device), expected,
                                   atol=1e-6, rtol=1e-6, msg=lambda msg: path + ': ' + msg)
    elif isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError(f'{path}: restored keys differ')
        for key in expected:
            assert_state_close(actual[key], expected[key], path + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise ValueError(f'{path}: restored sequence length differs')
        for index, value in enumerate(expected):
            assert_state_close(actual[index], value, f'{path}[{index}]')
    elif actual != expected:
        raise ValueError(f'{path}: restored value differs')


def capture(env):
    return {
        'schema_version': 1, 'num_envs': env.num_envs, 'seed': env.cfg.seed,
        'action_dim': env._action_dim,
        'scene': clone_tree(env.scene.get_state()),
        'tensors': {key: clone_tree(getattr(env, key)) for key in TENSORS},
        'values': {key: clone_tree(getattr(env, key)) for key in VALUES},
        'action_delay': clone_tree(env._action_delay.history),
        'episode_sums': clone_tree(env._episode_sums),
        'extras': clone_tree(env.extras),
        'generators': {key: getattr(env, key).get_state().clone()
                       for key in ('_goal_generator', '_command_generator')},
        'physics': {key: {prop: getattr(getattr(env, key).root_physx_view, 'get_' + prop)().clone()
                           for prop in PROPERTIES} for key in ('_robot', '_box')},
    }


def restore(env, state):
    for key, expected in (('schema_version', 1), ('num_envs', env.num_envs),
                          ('seed', env.cfg.seed), ('action_dim', env._action_dim)):
        if state.get(key) != expected:
            raise ValueError(f'training environment state mismatch: {key}')
    for key in TENSORS:
        if state['tensors'][key].shape != getattr(env, key).shape:
            raise ValueError(f'training tensor shape mismatch: {key}')
    # Initialize/reset public sensor caches, then restore physical and task state.
    env.scene.reset()
    indices = torch.arange(env.num_envs, dtype=torch.int32, device='cpu')
    for key in ('_robot', '_box'):
        view = getattr(env, key).root_physx_view
        for prop in PROPERTIES:
            target = getattr(view, 'get_' + prop)()
            getattr(view, 'set_' + prop)(state['physics'][key][prop].to(target.device), indices)
    env.scene.reset_to(state['scene'])
    for key in TENSORS:
        getattr(env, key).copy_(state['tensors'][key].to(env.device))
    for key in VALUES:
        if key == '_sim_step_counter' and key not in state['values']:
            setattr(env, key, state['values']['common_step_counter'] * env.cfg.decimation)
        else:
            setattr(env, key, clone_tree(state['values'][key]))
    env._action_delay.history.copy_(state['action_delay'].to(env.device))
    env._episode_sums = clone_tree(state['episode_sums'])
    env.extras = clone_tree(state['extras'])
    for key, value in state['generators'].items():
        getattr(env, key).set_state(value.cpu())
    env._robot.set_joint_effort_target(env._torques, joint_ids=env._all_joint_ids)
    env.scene.write_data_to_sim()
    restored = capture(env)
    if '_sim_step_counter' not in state['values']:
        restored['values'].pop('_sim_step_counter')
    assert_state_close(restored, state)
