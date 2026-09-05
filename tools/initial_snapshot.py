"""Shared initial-state recording; no changes to the legacy source tree."""

from __future__ import annotations

import numpy as np


def capture_legacy(env, path):
    import torch

    env.reset_idx(torch.arange(env.num_envs, device=env.device), start=True)
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
    arrays['body_names'] = np.array(env.body_names)
    masses, inertias, coms = [], [], []
    for handle in env.envs:
        properties = env.gym.get_actor_rigid_body_properties(handle, 0)
        masses.append([p.mass for p in properties])
        inertias.append([[p.inertia.x.x,p.inertia.x.y,p.inertia.x.z,
                          p.inertia.y.x,p.inertia.y.y,p.inertia.y.z,
                          p.inertia.z.x,p.inertia.z.y,p.inertia.z.z] for p in properties])
        coms.append([[p.com.x,p.com.y,p.com.z] for p in properties])
    arrays.update(masses=np.asarray(masses), inertias=np.asarray(inertias), coms=np.asarray(coms))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def apply_lab(env, path):
    import torch
    from dwbc_isaaclab.tasks.widow_go1.contracts import canonicalize_body_name

    with np.load(path) as snapshot:
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
        materials = view.get_material_properties().clone()
        friction = torch.as_tensor(snapshot['friction'],dtype=torch.float32).reshape(env.num_envs,1)
        materials[:,:,0] = friction.clamp_min(0)
        materials[:,:,1] = friction.clamp_min(0)
        view.set_material_properties(materials,ids)
        env._obs_history.zero_()
        env._action_delay.reset()
        env.episode_length_buf.zero_()
        env.scene.write_data_to_sim()
