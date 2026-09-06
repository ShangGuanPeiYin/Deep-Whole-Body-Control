"""Frozen custom PPO configuration ported from WidowGo1RoughCfgPPO."""


def dwbc_ppo_config(*, adaptive_arm_gains: bool = False) -> dict:
    action_dim = 24 if adaptive_arm_gains else 18
    init_std = [0.8, 1.0, 1.0] * 4 + [1.0] * 6
    min_policy_std = [0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3
    if adaptive_arm_gains:
        # Gain outputs are expressed directly in N*m/rad after the actor's
        # tenfold scale, so they need their own explicit exploration terms.
        init_std += [1.0] * 6
        min_policy_std += [0.05] * 6
    return {
        "policy": {
            "num_obs": 860,
            "num_actions": action_dim,
            "num_proprio": 76,
            "num_priv": 24,
            "history_len": 10,
            "init_std": init_std,
            "actor_hidden_dims": (128,),
            "critic_hidden_dims": (128,),
            "head_hidden_dims": (128, 128),
            "activation": "elu",
            "adaptive_arm_gains": adaptive_arm_gains,
            "adaptive_arm_gains_scale": 10.0,
        },
        "algorithm": {
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 2.0e-4,
            "gamma": 0.99,
            "lam": 0.95,
            "max_grad_norm": 1.0,
            "min_policy_std": min_policy_std,
            "mixing_schedule": (1.0, 0, 3000),
            "torque_supervision": False,
            "torque_supervision_schedule": (0.0, 1000, 1000),
            "dagger_update_freq": 20,
            "priv_reg_schedule": (0.0, 0.1, 3000, 7000),
        },
        "runner": {"num_steps_per_env": 40, "save_interval": 500},
    }
