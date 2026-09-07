"""Translate Isaac Lab transitions to the legacy DWBC runner protocol."""

from __future__ import annotations

import torch

from .contracts import ObservationLayout


def policy_observation(observations: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
    if isinstance(observations, torch.Tensor):
        result = observations
    elif "policy" in observations:
        result = observations["policy"]
    else:
        raise KeyError("Isaac Lab observation dictionary has no 'policy' entry")
    if result.ndim != 2 or result.shape[1] != ObservationLayout.flat:
        raise ValueError(f"policy observation must have shape (N,860), got {tuple(result.shape)}")
    return result


class LegacyRunnerAdapter:
    """Expose reset/step signatures consumed by the project-owned runner."""

    def __init__(self, env):
        self.env = env

    @property
    def num_envs(self):
        return self.env.num_envs

    @property
    def device(self):
        return self.env.device

    @property
    def action_dim(self):
        return self.env._action_dim

    def reset(self):
        observations, infos = self.env.reset()
        obs = policy_observation(observations)
        return obs, obs, infos

    def capture_training_state(self):
        from .training_state import capture
        return capture(self.env)

    def restore_training_state(self, state):
        from .training_state import restore
        restore(self.env, state)

    def arm_default_coefficients(self):
        if not self.env.cfg.torque_supervision:
            raise ValueError('PPO torque supervision requires environment torque_supervision=True')
        return (self.env._p_gains[12:18], self.env._d_gains[12:18],
                self.env._default_joint_pos[12:18])

    def update_command_curriculum(self):
        self.env.update_command_curriculum()

    def step(self, actions):
        observations, leg_reward, terminated, truncated, infos = self.env.step(actions)
        obs = policy_observation(observations)
        dones = terminated | truncated
        infos = dict(infos)
        infos.setdefault("leg_reward", leg_reward)
        infos.setdefault("time_outs", truncated)
        if "arm_reward" not in infos:
            raise KeyError("environment extras must contain arm_reward")
        return obs, obs, leg_reward, infos["arm_reward"], dones, infos
