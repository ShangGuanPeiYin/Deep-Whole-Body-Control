"""Minimal project-owned runner for the migrated DWBC PPO."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from dwbc_rsl_rl.algorithms import PPO
from dwbc_rsl_rl.modules import ActorCritic


CHECKPOINT_SCHEMA_VERSION = 1
CONTRACT = {
    "version": "dwbc-widow-go1-v1",
    "observation_dim": 860,
    "action_dim": 18,
    "value_channels": ["leg", "arm"],
}


def contract_for_policy(policy_config: dict) -> dict:
    """Derive the checkpoint contract from an explicit policy action branch."""
    adaptive = bool(policy_config.get("adaptive_arm_gains", False))
    action_dim = int(policy_config.get("num_actions", 18))
    if not adaptive and action_dim == 18:
        return dict(CONTRACT)
    if adaptive and action_dim == 24:
        return {
            "version": "dwbc-widow-go1-adaptive-gains-v1",
            "observation_dim": 860,
            "action_dim": 24,
            "value_channels": ["leg", "arm"],
        }
    raise ValueError(f"invalid DWBC policy action contract: adaptive_arm_gains={adaptive}, action_dim={action_dim}")


def make_checkpoint(iteration, model_state, optimizer_state, config, infos=None, *, contract=None):
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "contract": dict(CONTRACT if contract is None else contract),
        "iteration": int(iteration),
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer_state,
        "config": config,
        "infos": infos,
    }


def load_checkpoint(path: str | Path, map_location="cpu", *, expected_contract=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema mismatch: {checkpoint.get('schema_version')}")
    if checkpoint.get("contract") != (CONTRACT if expected_contract is None else expected_contract):
        raise ValueError(f"checkpoint contract mismatch: {checkpoint.get('contract')}")
    for key in ("model_state_dict", "optimizer_state_dict", "iteration"):
        if key not in checkpoint:
            raise ValueError(f"checkpoint is missing {key}")
    return checkpoint


class OnPolicyRunner:
    def __init__(self, env, config: dict, run_dir: str | Path, device="cuda:0"):
        self.env = env
        self.config = config
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        policy_cfg = config["policy"]
        self.actor_critic = ActorCritic(**policy_cfg).to(self.device)
        self.contract = contract_for_policy(policy_cfg)
        env_action_dim = getattr(env, "action_dim", self.actor_critic.num_actions)
        if env_action_dim != self.actor_critic.num_actions:
            raise ValueError(
                f"environment action_dim={env_action_dim} does not match policy action_dim={self.actor_critic.num_actions}"
            )
        self.algorithm = PPO(self.actor_critic, device=self.device, **config["algorithm"])
        if self.algorithm.torque_supervision:
            self.algorithm.set_arm_default_coeffs(*env.arm_default_coefficients())
        runner_cfg = config["runner"]
        self.num_steps_per_env = int(runner_cfg["num_steps_per_env"])
        self.save_interval = int(runner_cfg["save_interval"])
        self.current_iteration = 0
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.algorithm.init_storage(env.num_envs, self.num_steps_per_env)

    def save(self, path: str | Path | None = None):
        path = Path(path) if path is not None else self.run_dir / f"model_{self.current_iteration}.pt"
        checkpoint = make_checkpoint(
            self.current_iteration,
            self.actor_critic.state_dict(),
            self.algorithm.optimizer.state_dict(),
            self.config,
            contract=self.contract,
        )
        checkpoint['history_optimizer_state_dict'] = self.algorithm.hist_encoder_optimizer.state_dict()
        checkpoint['algorithm_counter'] = self.algorithm.counter
        checkpoint['torch_rng_state'] = torch.get_rng_state()
        if self.device.type == 'cuda':
            checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state(self.device)
        torch.save(checkpoint, path)
        return path

    def load(self, path: str | Path, load_optimizer=True):
        checkpoint = load_checkpoint(path, self.device, expected_contract=self.contract)
        self.actor_critic.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.algorithm.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if 'history_optimizer_state_dict' not in checkpoint:
                raise ValueError('checkpoint lacks history optimizer state required for resume')
            self.algorithm.hist_encoder_optimizer.load_state_dict(checkpoint['history_optimizer_state_dict'])
            self.algorithm.counter = int(checkpoint['algorithm_counter'])
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
            if self.device.type == 'cuda' and 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(),self.device)
        self.current_iteration = int(checkpoint["iteration"])
        return checkpoint.get("infos")

    def learn(self, num_learning_iterations: int):
        obs, critic_obs, _ = self.env.reset()
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        last_metrics = {}
        for iteration in range(self.current_iteration, self.current_iteration + num_learning_iterations):
            if hasattr(self.env, 'update_command_curriculum'):
                self.env.update_command_curriculum()
            started = time.perf_counter()
            use_history = iteration % self.algorithm.dagger_update_freq == 0
            leg_rewards = []
            arm_rewards = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.algorithm.act(obs, critic_obs, hist_encoding=use_history)
                    obs, critic_obs, leg_reward, arm_reward, dones, infos = self.env.step(actions)
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    leg_reward = leg_reward.to(self.device)
                    arm_reward = arm_reward.to(self.device)
                    dones = dones.to(self.device)
                    self.algorithm.process_env_step(leg_reward, arm_reward, dones, infos)
                    leg_rewards.append(leg_reward.mean())
                    arm_rewards.append(arm_reward.mean())
                self.algorithm.compute_returns(critic_obs)
            if use_history:
                hist_loss = self.algorithm.update_dagger()
                losses = {
                    "value_loss": 0.0, "surrogate_loss": 0.0, "torque_loss": 0.0,
                    "priv_reg_loss": 0.0, "value_mixing_ratio": 0.0,
                    "torque_weight": 0.0, "priv_reg_coef": 0.0,
                }
            else:
                hist_loss = 0.0
                losses = self.algorithm.update()
            elapsed = time.perf_counter() - started
            self.current_iteration = iteration + 1
            last_metrics = {
                "iteration": iteration,
                "leg_reward": torch.stack(leg_rewards).mean().item(),
                "arm_reward": torch.stack(arm_rewards).mean().item(),
                "history_latent_loss": hist_loss,
                "fps": self.num_steps_per_env * self.env.num_envs / max(elapsed, 1e-9),
                **losses,
            }
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            print(json.dumps(last_metrics, sort_keys=True))
            if self.current_iteration % self.save_interval == 0:
                self.save()
        checkpoint_path = self.save()
        return last_metrics, checkpoint_path

    def get_inference_policy(self, use_history=True):
        self.actor_critic.eval()
        return lambda observations: self.actor_critic.act_inference(observations, hist_encoding=use_history)
