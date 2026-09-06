"""Shared initial-state recording; no changes to the legacy source tree."""

from __future__ import annotations

import numpy as np


def sanitize_state_only_joint_positions(positions, joint_names, lower, upper, state_only_names):
    """Clamp only named non-actuated state joints into their declared interval."""
    result = np.asarray(positions).copy()
    if result.ndim != 2 or result.shape[1] != len(joint_names):
        raise ValueError('joint positions must have shape (N, len(joint_names))')
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    if lower.shape != (len(joint_names),) or upper.shape != lower.shape:
        raise ValueError('joint limit shape does not match joint names')
    indices = {name: index for index, name in enumerate(joint_names)}
    missing = set(state_only_names) - set(indices)
    if missing:
        raise ValueError(f'state-only joints missing from snapshot: {sorted(missing)}')
    for name in state_only_names:
        index = indices[name]
        result[:, index] = np.clip(result[:, index], lower[index], upper[index])
    return result


def validate_physics_snapshot(snapshot, num_envs):
    if 'body_names' not in snapshot:
        raise ValueError('missing physics snapshot field: body_names')
    bodies = len(snapshot['body_names'])
    shapes = dict(masses=(num_envs, bodies), inertias=(num_envs, bodies, 9),
                  coms=(num_envs, bodies, 3), box_masses=(num_envs, 1),
                  box_inertias=(num_envs, 9), box_coms=(num_envs, 3),
                  box_materials=(num_envs, 1, 3))
    for name, shape in shapes.items():
        if name not in snapshot:
            raise ValueError(f'missing physics snapshot field: {name}; export a complete snapshot')
        if snapshot[name].shape != shape or not np.isfinite(snapshot[name]).all():
            raise ValueError(f'invalid physics snapshot field: {name}; expected finite shape {shape}')


def capture_legacy(env, path, *, sanitize_state_only_joint_limits=False):
    import torch
    from isaacgym import gymtorch

    env.reset_idx(torch.arange(env.num_envs, device=env.device), start=True)
    if sanitize_state_only_joint_limits:
        dof_properties = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
        position = sanitize_state_only_joint_positions(
            env.dof_pos.detach().cpu().numpy(), env.dof_names,
            dof_properties['lower'], dof_properties['upper'],
            ('widow_left_finger', 'widow_right_finger'),
        )
        env.dof_pos.copy_(torch.as_tensor(position, device=env.device))
        env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
        env.gym.refresh_dof_state_tensor(env.sim)
    fields = {
        'root_state': env.root_states,
        'box_state': env.box_root_state,
        'dof_pos': env.ig2raisim(env.dof_pos),
        'dof_vel': env.ig2raisim(env.dof_vel),
        'commands': env.commands,
        'goal_start': env.ee_start_sphere,
        'goal_end': env.ee_goal_sphere,
        'goal_timer': env.goal_timer,
        'trajectory_steps': env.traj_timesteps,
        'trajectory_total_steps': env.traj_total_timesteps,
        'mass_params': env.mass_params_tensor,
        'friction': env.friction_coeffs_tensor,
        'motor_strength': env.ig2raisim_wo_gripper(env.motor_strength),
    }
    arrays = {key: value.detach().cpu().numpy().copy() for key,value in fields.items()}
    # Root writes happen after the old task's last rigid-state refresh. Refresh
    # only for this detached diagnostic capture; this does not simulate a step.
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    arrays['body_state'] = env.rigid_body_state.detach().cpu().numpy().copy()
    arrays['body_names'] = np.array(env.body_names)
    dof_properties = env.gym.get_actor_dof_properties(env.envs[0], env.actor_handles[0])
    arrays['native_joint_names'] = np.array(env.dof_names)
    for name in dof_properties.dtype.names:
        arrays['dof_property_' + name] = dof_properties[name].copy()
    masses, inertias, coms = [], [], []
    for handle in env.envs:
        properties = env.gym.get_actor_rigid_body_properties(handle, 0)
        masses.append([p.mass for p in properties])
        inertias.append([[p.inertia.x.x,p.inertia.x.y,p.inertia.x.z,
                          p.inertia.y.x,p.inertia.y.y,p.inertia.y.z,
                          p.inertia.z.x,p.inertia.z.y,p.inertia.z.z] for p in properties])
        coms.append([[p.com.x,p.com.y,p.com.z] for p in properties])
    arrays.update(masses=np.asarray(masses), inertias=np.asarray(inertias), coms=np.asarray(coms))
    box_masses, box_inertias, box_coms, box_materials = [], [], [], []
    for handle in env.envs:
        prop = env.gym.get_actor_rigid_body_properties(handle, 1)[0]
        box_masses.append([prop.mass])
        box_inertias.append([getattr(getattr(prop.inertia, a), b) for a in 'xyz' for b in 'xyz'])
        box_coms.append([prop.com.x, prop.com.y, prop.com.z])
        material = env.gym.get_actor_rigid_shape_properties(handle, 1)[0]
        box_materials.append([[material.friction, material.friction, material.restitution]])
    arrays.update(box_masses=np.asarray(box_masses), box_inertias=np.asarray(box_inertias),
                  box_coms=np.asarray(box_coms), box_materials=np.asarray(box_materials))
    validate_physics_snapshot(arrays, env.num_envs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def apply_lab(env, path):
    import torch
    from dwbc_isaaclab.tasks.widow_go1.contracts import canonicalize_body_name

    with np.load(path) as snapshot:
        validate_physics_snapshot(snapshot, env.num_envs)
        tensor = lambda key: torch.as_tensor(snapshot[key].copy(), device=env.device, dtype=torch.float32)
        root = tensor('root_state')
        if root.shape != (env.num_envs,13):
            raise ValueError('initial snapshot environment count mismatch')
        pose = torch.cat((root[:,:3],root[:,6:7],root[:,3:6]),-1)
        env._robot.write_root_pose_to_sim(pose)
        env._robot.write_root_velocity_to_sim(root[:,7:])
        env._robot.write_joint_state_to_sim(tensor('dof_pos'),tensor('dof_vel'),joint_ids=env._all_joint_ids)
        box = tensor('box_state')
        env._box.write_root_pose_to_sim(torch.cat((box[:,:3],box[:,6:7],box[:,3:6]),-1))
        env._box.write_root_velocity_to_sim(box[:,7:])
        for target,source in (
            ('_commands','commands'),('_ee_start_sphere','goal_start'),('_ee_goal_sphere','goal_end'),
            ('_goal_timer','goal_timer'),('_trajectory_steps','trajectory_steps'),
            ('_trajectory_total_steps','trajectory_total_steps'),('_mass_params','mass_params'),
            ('_friction_coefficients','friction'),('_motor_strength','motor_strength')):
            getattr(env,target).copy_(tensor(source).reshape_as(getattr(env,target)))
        names = list(snapshot['body_names'])
        order = [names.index(canonicalize_body_name(name)) for name in env._robot.body_names]
        ids = torch.arange(env.num_envs,dtype=torch.int32)
        view = env._robot.root_physx_view
        view.set_masses(torch.as_tensor(snapshot['masses'][:,order],dtype=torch.float32),ids)
        view.set_inertias(torch.as_tensor(snapshot['inertias'][:,order],dtype=torch.float32),ids)
        coms = view.get_coms().clone()
        coms[:,:,:3] = torch.as_tensor(snapshot['coms'][:,order],dtype=torch.float32)
        view.set_coms(coms,ids)
        box_view = env._box.root_physx_view
        box_view.set_masses(torch.as_tensor(snapshot['box_masses'], dtype=torch.float32), ids)
        box_view.set_inertias(torch.as_tensor(snapshot['box_inertias'], dtype=torch.float32), ids)
        box_coms = box_view.get_coms().clone()
        box_coms[:, :3] = torch.as_tensor(snapshot['box_coms'], dtype=torch.float32)
        box_view.set_coms(box_coms, ids)
        box_view.set_material_properties(torch.as_tensor(snapshot['box_materials'], dtype=torch.float32), ids)
        materials = view.get_material_properties().clone()
        friction = torch.as_tensor(snapshot['friction'],dtype=torch.float32).reshape(env.num_envs,1)
        materials[:,:,0] = friction.clamp_min(0)
        materials[:,:,1] = friction.clamp_min(0)
        view.set_material_properties(materials,ids)
        env._obs_history.zero_()
        env._action_delay.reset()
        env.episode_length_buf.zero_()
        env.scene.write_data_to_sim()
