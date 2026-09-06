"""Project-owned actor-critic preserving the original DWBC network semantics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "lrelu": nn.LeakyReLU, "selu": nn.SELU, "tanh": nn.Tanh}
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation: {name}") from exc


def _mlp(widths: Sequence[int], activation: str, *, final_tanh: bool = False, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            layers.append(_activation(activation))
        elif final_tanh:
            layers.append(nn.Tanh())
        elif final_activation:
            layers.append(_activation(activation))
    return nn.Sequential(*layers)


class StateHistoryEncoder(nn.Module):
    """The original 10-frame projection plus temporal Conv1d encoder."""

    def __init__(self, input_size: int = 76, history_len: int = 10, output_size: int = 20, activation: str = "elu"):
        super().__init__()
        if history_len != 10:
            raise ValueError("the frozen DWBC contract requires exactly 10 history frames")
        self.input_size = input_size
        self.history_len = history_len
        self.projection = nn.Sequential(nn.Linear(input_size, 30), _activation(activation))
        self.temporal = nn.Sequential(
            nn.Conv1d(30, 20, kernel_size=4, stride=2), _activation(activation),
            nn.Conv1d(20, 10, kernel_size=2, stride=1), _activation(activation), nn.Flatten(),
        )
        self.output = nn.Sequential(nn.Linear(30, output_size), _activation(activation))

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.history_len, self.input_size):
            raise ValueError(f"history must have shape (N,{self.history_len},{self.input_size}), got {tuple(history.shape)}")
        batch = history.shape[0]
        projected = self.projection(history.reshape(batch * self.history_len, self.input_size))
        encoded = self.temporal(projected.reshape(batch, self.history_len, 30).transpose(1, 2))
        return self.output(encoded)


class _Actor(nn.Module):
    def __init__(self, num_proprio, num_priv, history_len, actor_hidden_dims, head_hidden_dims,
                 activation, adaptive_arm_gains, adaptive_arm_gains_scale):
        super().__init__()
        self.num_proprio = num_proprio
        self.num_priv = num_priv
        self.history_len = history_len
        self.adaptive_arm_gains = adaptive_arm_gains
        self.adaptive_arm_gains_scale = adaptive_arm_gains_scale
        self.priv_encoder = _mlp((num_priv, 64, 20), activation, final_activation=True)
        self.history_encoder = StateHistoryEncoder(num_proprio, history_len, 20, activation)
        self.backbone = _mlp((num_proprio + 20, *actor_hidden_dims), activation, final_activation=True)
        backbone_width = actor_hidden_dims[-1] if actor_hidden_dims else num_proprio + 20
        self.leg_head = _mlp((backbone_width, *head_hidden_dims, 12), activation, final_tanh=True)
        arm_width = 12 if adaptive_arm_gains else 6
        self.arm_head = _mlp((backbone_width, *head_hidden_dims, arm_width), activation, final_tanh=True)

    def infer_priv_latent(self, obs: torch.Tensor) -> torch.Tensor:
        return self.priv_encoder(obs[:, self.num_proprio:self.num_proprio + self.num_priv])

    def infer_hist_latent(self, obs: torch.Tensor) -> torch.Tensor:
        history = obs[:, -self.history_len * self.num_proprio:]
        return self.history_encoder(history.reshape(-1, self.history_len, self.num_proprio))

    def forward(self, obs: torch.Tensor, use_history: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        proprio = obs[:, :self.num_proprio]
        latent = self.infer_hist_latent(obs) if use_history else self.infer_priv_latent(obs)
        features = self.backbone(torch.cat((proprio, latent), dim=-1))
        legs = self.leg_head(features)
        arm_raw = self.arm_head(features)
        if self.adaptive_arm_gains:
            arm_action = arm_raw[:, :6]
            gain_delta = arm_raw[:, 6:] * self.adaptive_arm_gains_scale
        else:
            arm_action = arm_raw
            gain_delta = torch.zeros_like(arm_action)
        return torch.cat((legs, arm_action), dim=-1), gain_delta


class ActorCritic(nn.Module):
    """Dual-value DWBC policy with explicit fixed- and adaptive-gain contracts."""

    is_recurrent = False

    def __init__(self, num_obs=860, num_actions=18, num_proprio=76, num_priv=24, history_len=10, *,
                 actor_hidden_dims=(128, 128, 128), critic_hidden_dims=(128, 128, 128),
                 head_hidden_dims=(64,), activation="elu", init_std=1.0,
                 adaptive_arm_gains=False, adaptive_arm_gains_scale=10.0):
        super().__init__()
        if (num_obs, num_proprio, num_priv, history_len) != (860, 76, 24, 10):
            raise ValueError("DWBC contract is fixed at obs/proprio/priv/history = 860/76/24/10")
        expected_actions = 24 if adaptive_arm_gains else 18
        if num_actions != expected_actions:
            raise ValueError(
                f"adaptive_arm_gains={adaptive_arm_gains} requires action width {expected_actions}, got {num_actions}"
            )
        self.num_actions = num_actions
        self.num_leg_actions = 12
        self.num_arm_actions = 6
        self.actor = _Actor(num_proprio, num_priv, history_len, actor_hidden_dims, head_hidden_dims,
                            activation, adaptive_arm_gains, adaptive_arm_gains_scale)
        self.critic_backbone = _mlp((num_proprio + num_priv, *critic_hidden_dims), activation, final_activation=True)
        critic_width = critic_hidden_dims[-1] if critic_hidden_dims else num_proprio + num_priv
        self.critic_leg_head = _mlp((critic_width, *head_hidden_dims, 1), activation)
        self.critic_arm_head = _mlp((critic_width, *head_hidden_dims, 1), activation)
        initial_std = torch.as_tensor(init_std, dtype=torch.float32).flatten()
        if initial_std.numel() == 1:
            initial_std = initial_std.repeat(num_actions)
        if initial_std.numel() != num_actions:
            raise ValueError(f"init_std must be scalar or width {num_actions}, got {initial_std.numel()}")
        self.std = nn.Parameter(initial_std.clone())
        self.distribution: Normal | None = None
        self.arm_gain_delta = torch.empty(0, 6)

    def infer_priv_latent(self, obs):
        return self.actor.infer_priv_latent(obs)

    def infer_history_latent(self, obs):
        return self.actor.infer_hist_latent(obs)

    def action_mean_and_gains(self, observations, use_history=False):
        actions, gains = self.actor(observations, use_history)
        if self.actor.adaptive_arm_gains:
            return torch.cat((actions, gains), dim=-1), gains
        return actions, gains

    def evaluate(self, critic_observations, **_):
        features = self.critic_backbone(critic_observations[:, :100])
        return torch.cat((self.critic_leg_head(features), self.critic_arm_head(features)), dim=-1)

    def forward(self, observations, use_history=False):
        mean, self.arm_gain_delta = self.action_mean_and_gains(observations, use_history)
        return mean, self.evaluate(observations)

    def update_distribution(self, observations, use_history=False):
        mean, self.arm_gain_delta = self.action_mean_and_gains(observations, use_history)
        self.distribution = Normal(mean, self.std.clamp_min(1e-6).expand_as(mean), validate_args=False)

    def act(self, observations, hist_encoding=False, **_):
        self.update_distribution(observations, hist_encoding)
        return self.distribution.sample()

    def act_inference(self, observations, hist_encoding=False):
        mean, self.arm_gain_delta = self.action_mean_and_gains(observations, hist_encoding)
        return mean

    def get_actions_log_prob(self, actions):
        if self.distribution is None:
            raise RuntimeError("call act() before requesting log probability")
        elementwise = self.distribution.log_prob(actions)
        return torch.stack((elementwise[:, :12].sum(-1), elementwise[:, 12:].sum(-1)), dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        elementwise = self.distribution.entropy()
        return torch.stack((elementwise[:, :12].sum(-1), elementwise[:, 12:].sum(-1)), dim=-1)

    def reset(self, dones=None):
        del dones
