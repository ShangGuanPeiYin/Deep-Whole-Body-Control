"""DirectRLEnv implementation of the legacy WidowGo1 control lifecycle."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

from .contracts import POLICY_ACTION_NAMES, ROBOT_JOINT_NAMES, validate_joint_names
from .control import ActionDelayBuffer, compute_pd_torques, compute_osc_torques
from .commands import sample_commands, sample_box_offsets
from .goals import cart_to_sphere, sphere_to_cart, goal_collision_mask, orientation_error_xyzw, wxyz_to_xyzw
from .observations import (build_legacy_observation, compose_proprioception,
                           build_privileged_observation, relative_joint_positions)
from .resets import DEFAULT_JOINT_POS, sample_reset_state
from .rewards import arm_reward, combine_rewards, leg_reward, termination_flags
from .widow_go1_env_cfg import WidowGo1EnvCfg


EFFORT_LIMITS = torch.tensor(
    [23.7] * 12 + [10.0, 20.0, 15.0, 2.0, 5.0, 1.0, 5.0, 5.0], dtype=torch.float32
)


class WidowGo1Env(DirectRLEnv):
    cfg: WidowGo1EnvCfg

    def __init__(self, cfg: WidowGo1EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Collision shapes live in instance proxies, which USD spawn overrides
        # cannot edit. Set the legacy offsets on the actual PhysX shapes.
        physics_view = self._robot.root_physx_view
        shape_ids = torch.arange(self.num_envs, dtype=torch.int32)
        physics_view.set_contact_offsets(
            torch.full_like(physics_view.get_contact_offsets(), cfg.robot.spawn.collision_props.contact_offset),
            shape_ids,
        )
        physics_view.set_rest_offsets(
            torch.full_like(physics_view.get_rest_offsets(), cfg.robot.spawn.collision_props.rest_offset),
            shape_ids,
        )
        validate_joint_names(self._robot.joint_names, ROBOT_JOINT_NAMES)
        self._all_joint_ids, resolved = self._robot.find_joints(list(ROBOT_JOINT_NAMES), preserve_order=True)
        if tuple(resolved) != ROBOT_JOINT_NAMES:
            raise RuntimeError(f"canonical joint resolution failed: {resolved}")
        self._policy_joint_ids = self._all_joint_ids[:18]
        self._gripper_joint_ids = self._all_joint_ids[18:]
        self._feet_ids, feet = self._robot.find_bodies(
            ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True
        )
        if tuple(feet) != ("FR_foot", "FL_foot", "RR_foot", "RL_foot"):
            raise RuntimeError(f"foot resolution failed: {feet}")
        self._contact_feet_ids, contact_feet = self._contact_sensor.find_bodies(
            ["FR_foot", "FL_foot", "RR_foot", "RL_foot"], preserve_order=True)
        if tuple(contact_feet) != tuple(feet):
            raise RuntimeError(f"contact sensor foot resolution failed: {contact_feet}")
        ee_ids, ee_names = self._robot.find_bodies("wx250s_ee_gripper_link")
        if len(ee_ids) != 1:
            raise RuntimeError(f"end-effector resolution failed: {ee_names}")
        self._ee_id = ee_ids[0]
        self._osc_body_ids, _ = self._robot.find_bodies(
            ["wx250s_" + name + "_link" for name in (
                "shoulder", "upper_arm", "upper_forearm", "lower_forearm", "wrist",
                "gripper", "ee_gripper", "left_finger", "right_finger",
            )], preserve_order=True,
        )
        if len(self._osc_body_ids) != 9:
            raise RuntimeError("OSC gravity compensation requires all nine arm bodies")
        base_ids, _ = self._robot.find_bodies("base")
        self._base_id = base_ids[0]

        self._action_delay = ActionDelayBuffer(self.num_envs, 18, cfg.action_delay, self.device)
        self._raw_actions = torch.zeros(self.num_envs, 18, device=self.device)
        self._actions = torch.zeros_like(self._raw_actions)
        self._last_actions = torch.zeros_like(self._raw_actions)
        self._torques = torch.zeros(self.num_envs, 20, device=self.device)
        self._default_joint_pos = DEFAULT_JOINT_POS.to(self.device)
        self._action_scale = torch.tensor(cfg.action_scale, device=self.device)
        self._p_gains = torch.tensor(cfg.p_gains, device=self.device)
        self._d_gains = torch.tensor(cfg.d_gains, device=self.device)
        self._effort_limits = EFFORT_LIMITS.to(self.device)
        self._apply_startup_randomization()
        self._obs_history = torch.zeros(self.num_envs, 10, 76, device=self.device)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self._current_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_goal_delta_orientation = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_timer = torch.zeros(self.num_envs, device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 101)
        self._trajectory_steps = 50.0 + 100.0 * torch.rand(
            self.num_envs, generator=generator, device=self.device
        )
        self._trajectory_total_steps = self._trajectory_steps + 25.0 + 75.0 * torch.rand(
            self.num_envs, generator=generator, device=self.device
        )
        self._goal_generator = generator
        self._command_generator = torch.Generator(device=self.device).manual_seed(cfg.seed + 102)
        self._box_offsets = sample_box_offsets(self.num_envs, self._command_generator, self.device)
        self._curriculum_counter = 0
        self._lin_range = (0.0, 0.0)
        self._yaw_range = (0.0, 0.0)
        self._goal_ranges = ((0.6, 0.6), (torch.pi / 4, torch.pi / 4), (-torch.pi / 6, torch.pi / 6))
        self._episode_sums: dict[str, torch.Tensor] = {}
        self._reset_counter = 0

    def _apply_startup_randomization(self) -> None:
        """Apply legacy startup distributions and retain their exact privileged values."""
        cpu_generator = torch.Generator(device="cpu").manual_seed(self.cfg.seed + 17)
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device="cpu")
        masses = self._robot.root_physx_view.get_masses().clone()
        inertias = self._robot.root_physx_view.get_inertias().clone()
        base_delta = -0.5 + 3.0 * torch.rand(self.num_envs, generator=cpu_generator)
        gripper_delta = 0.1 * torch.rand(self.num_envs, generator=cpu_generator)
        for body_id, delta in ((self._base_id, base_delta), (self._ee_id, gripper_delta)):
            before = masses[:, body_id].clone()
            masses[:, body_id] += delta
            inertias[:, body_id] *= (masses[:, body_id] / before).unsqueeze(-1)
        self._robot.root_physx_view.set_masses(masses, env_ids)
        self._robot.root_physx_view.set_inertias(inertias, env_ids)

        com_delta = -0.15 + 0.3 * torch.rand(self.num_envs, 3, generator=cpu_generator)
        coms = self._robot.root_physx_view.get_coms().clone()
        coms[:, self._base_id, :3] += com_delta
        self._robot.root_physx_view.set_coms(coms, env_ids)

        friction = torch.clamp(-0.5 + 3.5 * torch.rand(self.num_envs, generator=cpu_generator), min=0.0)
        materials = self._robot.root_physx_view.get_material_properties().clone()
        materials[:, :, 0] = friction[:, None]
        materials[:, :, 1] = friction[:, None]
        materials[:, :, 2] = 0.0
        self._robot.root_physx_view.set_material_properties(materials, env_ids)

        box_masses = self._box.root_physx_view.get_masses().clone()
        box_inertias = self._box.root_physx_view.get_inertias().clone()
        box_delta = -0.001 + 0.051 * torch.rand(self.num_envs, generator=cpu_generator)
        box_before = box_masses[:, 0].clone()
        box_masses[:, 0] += box_delta
        # RigidObjectView returns (N,9), unlike ArticulationView's (N,B,9).
        box_inertias *= (box_masses[:, 0] / box_before).unsqueeze(-1)
        self._box.root_physx_view.set_masses(box_masses, env_ids)
        self._box.root_physx_view.set_inertias(box_inertias, env_ids)

        self._mass_params = torch.cat(
            (base_delta[:, None], com_delta, gripper_delta[:, None]), dim=-1
        ).to(self.device)
        self._friction_coefficients = friction[:, None].to(self.device)
        self._motor_strength = torch.cat(
            (
                0.7 + 0.6 * torch.rand(self.num_envs, 12, generator=cpu_generator),
                0.7 + 0.6 * torch.rand(self.num_envs, 6, generator=cpu_generator),
            ),
            dim=-1,
        ).to(self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._box = RigidObject(self.cfg.box)
        self.scene.rigid_objects["box"] = self._box
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )
        generator = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        self._env_origins = torch.zeros(self.scene.num_envs, 3, device=self.device)
        self._env_origins[:, 0] = -3.75 + 0.75 * torch.rand(
            self.scene.num_envs, generator=generator, device=self.device
        )
        self._env_origins[:, 1] = -15.0 + 30.0 * torch.rand(
            self.scene.num_envs, generator=generator, device=self.device
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        expected = (self.num_envs, 18)
        if tuple(actions.shape) != expected:
            raise ValueError(f"expected actions shape {expected}, got {tuple(actions.shape)}")
        self._raw_actions = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions).clone()
        self._last_actions.copy_(self._actions)
        self._actions = self._action_delay.push(self._raw_actions)
        self._control_substep = 0

    def _apply_action(self):
        joint_pos = self._robot.data.joint_pos[:, self._all_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self._all_joint_ids]
        self._torques = compute_pd_torques(
            self._actions,
            joint_pos,
            joint_vel,
            default_joint_pos=self._default_joint_pos,
            action_scale=self._action_scale,
            motor_strength=self._motor_strength,
            p_gains=self._p_gains,
            d_gains=self._d_gains,
            effort_limits=self._effort_limits,
        )
        self._robot.set_joint_effort_target(self._torques, joint_ids=self._all_joint_ids)
        if self.cfg.torque_supervision and self._control_substep == 0:
            self.extras["target_arm_torques"] = self._compute_arm_osc_target()
            self.extras["current_arm_dof_pos"] = joint_pos[:, 12:18].clone()
            self.extras["current_arm_dof_vel"] = joint_vel[:, 12:18].clone()
        self._control_substep += 1

    def _compute_arm_osc_target(self):
        view = self._robot.root_physx_view
        root_dofs = 0 if self._robot.is_fixed_base else 6
        arm_columns = [index + root_dofs for index in self._all_joint_ids[12:18]]
        jacobians = view.get_jacobians()[..., arm_columns]
        body_shift = 1 if self._robot.is_fixed_base else 0
        ee_jacobian = jacobians[:, self._ee_id - body_shift]
        body_jacobians = jacobians[:, [index - body_shift for index in self._osc_body_ids]]
        mass = view.get_generalized_mass_matrices()[:, arm_columns, :][:, :, arm_columns]
        # The original controller uses environment zero's link masses for every
        # environment, even when gripper mass is randomized. Preserve that rule.
        masses = view.get_masses()[:1, self._osc_body_ids].to(self.device).expand(self.num_envs, -1)
        reference = self._robot.data.root_pos_w.clone()
        reference[:, 2] = 0.53
        goal = reference + math_utils.quat_apply(
            math_utils.yaw_quat(self._robot.data.root_quat_w), sphere_to_cart(self._current_ee_goal_sphere)
        )
        position_error = goal - self._robot.data.body_pos_w[:, self._ee_id]
        current = wxyz_to_xyzw(self._robot.data.body_quat_w[:, self._ee_id])
        current = current / torch.linalg.vector_norm(current, dim=-1, keepdim=True)
        desired = current.new_tensor([0., 0.7071068, 0., 0.7071068]).expand_as(current)
        error = torch.cat((position_error, orientation_error_xyzw(desired, current)), dim=-1)
        kp = current.new_tensor([100., 100., 100., 30., 30., 30.])
        return compute_osc_torques(
            mass, ee_jacobian, error, self._robot.data.body_vel_w[:, self._ee_id],
            kp, 2 * torch.sqrt(kp), body_jacobians, masses,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        root_quat = self._robot.data.root_quat_w
        roll, pitch, _ = math_utils.euler_xyz_from_quat(root_quat)
        joint_pos = self._robot.data.joint_pos[:, self._all_joint_ids].clone()
        joint_vel = self._robot.data.joint_vel[:, self._all_joint_ids]
        net_forces = self._contact_sensor.data.net_forces_w
        feet_contacts = (torch.linalg.vector_norm(net_forces[:, self._contact_feet_ids], dim=-1) > 1.5).float()
        proprio = compose_proprioception(
            orientation=torch.stack((roll, pitch), dim=-1),
            angular_velocity=self._robot.data.root_ang_vel_b,
            dof_pos=relative_joint_positions(joint_pos, self._default_joint_pos),
            dof_vel=joint_vel * 0.05,
            previous_action=self._raw_actions,
            feet_contacts=feet_contacts,
            command=self._commands,
            ee_goal=self._current_ee_goal_sphere,
            ee_orientation_error=self._ee_goal_delta_orientation,
        )
        privileged = build_privileged_observation(
            self._mass_params, self._friction_coefficients, self._motor_strength
        )
        observation = build_legacy_observation(proprio, privileged, self._obs_history)
        fresh = self.episode_length_buf <= 1
        shifted = torch.cat((self._obs_history[:, 1:], proprio[:, None]), dim=1)
        repeated = proprio[:, None].repeat(1, 10, 1)
        self._obs_history = torch.where(fresh[:, None, None], repeated, shifted)
        return {"policy": torch.clamp(observation, -100.0, 100.0)}

    def _get_rewards(self) -> torch.Tensor:
        joint_vel = self._robot.data.joint_vel[:, self._all_joint_ids]
        forces = self._contact_sensor.data.net_forces_w[:, self._contact_feet_ids]
        leg, leg_terms = leg_reward(
            actions=self._actions,
            torques=self._torques,
            joint_vel=joint_vel,
            base_lin_vel=self._robot.data.root_lin_vel_b,
            base_ang_vel=self._robot.data.root_ang_vel_b,
            commands=self._commands,
            foot_force_z=forces[:, :, 2],
        )
        yaw_quaternion = math_utils.yaw_quat(self._robot.data.root_quat_w)
        reference = torch.cat(
            (
                self._robot.data.root_pos_w[:, :2],
                torch.full((self.num_envs, 1), 0.53, device=self.device),
            ),
            dim=-1,
        )
        ee_world = self._robot.data.body_pos_w[:, self._ee_id]
        ee_local = math_utils.quat_apply_inverse(yaw_quaternion, ee_world - reference)
        arm, arm_terms = arm_reward(
            ee_position_local=ee_local,
            ee_goal_sphere=self._current_ee_goal_sphere,
            torques=self._torques,
            joint_vel=joint_vel,
        )
        reward, channels = combine_rewards(leg, arm)
        self.extras.update(channels)
        self.extras["reward_terms"] = {**leg_terms, **arm_terms}
        for name, value in self.extras["reward_terms"].items():
            self._episode_sums.setdefault(name, torch.zeros_like(value)).add_(value)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._advance_goals()
        command_ids = (self.episode_length_buf % 150 == 0).nonzero().flatten()
        self._commands[command_ids] = sample_commands(
            len(command_ids), self._command_generator, self.device, self._lin_range, self._yaw_range
        )
        if self.common_step_counter % 150 == 0:
            velocity = self._robot.data.root_vel_w.clone()
            pushes = torch.rand(self.num_envs, 2, generator=self._command_generator, device=self.device) - 0.5
            velocity[:, :2] = torch.where((self._commands.sum(-1) == 0)[:, None], pushes * 2.5, pushes)
            self._robot.write_root_velocity_to_sim(velocity)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self._robot.data.root_quat_w)
        terminated, reason = termination_flags(
            roll, pitch, self._robot.data.root_pos_w[:, 2], self._current_ee_goal_sphere
        )
        time_out = self.episode_length_buf > self.max_episode_length
        reason[time_out] = 4
        self.extras["termination_reason"] = reason
        self.extras["time_outs"] = time_out
        return terminated, time_out

    def _sample_goals(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._ee_start_sphere[env_ids] = self._ee_goal_sphere[env_ids]
        pending = env_ids
        for _ in range(10):
            for axis, (lower, upper) in enumerate(self._goal_ranges):
                self._ee_goal_sphere[pending, axis] = lower + (upper - lower) * torch.rand(
                    len(pending), generator=self._goal_generator, device=self.device)
            pending = pending[goal_collision_mask(self._ee_start_sphere[pending], self._ee_goal_sphere[pending])]
            if len(pending) == 0:
                break
        self._goal_timer[env_ids] = 0.0

    def _advance_goals(self) -> None:
        interpolation = torch.clamp(self._goal_timer / self._trajectory_steps, 0.0, 1.0)
        self._current_ee_goal_sphere = torch.lerp(
            self._ee_start_sphere, self._ee_goal_sphere, interpolation[:, None]
        )
        self._goal_timer += 1.0
        self._sample_goals((self._goal_timer > self._trajectory_total_steps).nonzero().flatten())

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        state = sample_reset_state(
            len(env_ids),
            self.cfg.seed + self._reset_counter,
            self.device,
            origins=self._env_origins[env_ids],
            default_joint_pos=self._default_joint_pos,
            origin_perturb=self.cfg.origin_perturb_range,
            velocity_perturb=self.cfg.init_velocity_perturb_range,
        )
        self._reset_counter += 1
        self._robot.write_root_pose_to_sim(state.root_pose, env_ids)
        self._robot.write_root_velocity_to_sim(state.root_velocity, env_ids)
        self._robot.write_joint_state_to_sim(
            state.joint_position, state.joint_velocity, joint_ids=self._all_joint_ids, env_ids=env_ids
        )
        box_pose = torch.zeros(len(env_ids), 7, device=self.device)
        box_pose[:, 0] = 0.0
        box_pose[:, 1] = state.root_pose[:, 1] + self._box_offsets[env_ids]
        box_pose[:, 2] = 0.21
        box_pose[:, 3] = 1.0
        self._box.write_root_pose_to_sim(box_pose, env_ids)
        self._box.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids)
        self._action_delay.reset(env_ids)
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._obs_history[env_ids] = 0.0
        # Legacy falls retain commands; startup and timeouts resample them.
        if self._reset_counter == 1:
            command_ids = env_ids
        else:
            command_ids = env_ids[self.reset_time_outs[env_ids]]
        self._commands[command_ids] = sample_commands(
            len(command_ids), self._command_generator, self.device, self._lin_range, self._yaw_range)
        self.extras['episode'] = {}
        for name, values in self._episode_sums.items():
            self.extras['episode']['rew_' + name] = values[env_ids].mean() / self.cfg.episode_length_s
            values[env_ids] = 0
        self._sample_goals(env_ids)

    def update_command_curriculum(self):
        self._curriculum_counter += 1
        # Frozen legacy schedules are all [0,1], reaching final bounds on update 1.
        self._lin_range = (0.0, 0.9)
        self._yaw_range = (-1.0, 1.0)
        self._goal_ranges = ((0.2, 0.7), (-2 * torch.pi / 5, torch.pi / 5), (-3 * torch.pi / 5, 3 * torch.pi / 5))
