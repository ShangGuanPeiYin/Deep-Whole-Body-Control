"""Frozen custom PPO configuration ported from WidowGo1RoughCfgPPO."""


def dwbc_ppo_config() -> dict:
    return {
        "policy": {
            "num_obs": 860,
            "num_actions": 18,
            "num_proprio": 76,
            "num_priv": 24,
            "history_len": 10,
            "init_std": [0.8, 1.0, 1.0] * 4 + [1.0] * 6,
            "actor_hidden_dims": (128,),
            "critic_hidden_dims": (128,),
            "head_hidden_dims": (128, 128),
            "activation": "elu",
            "adaptive_arm_gains": False,
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
            "min_policy_std": [0.15, 0.25, 0.25] * 4 + [0.2] * 3 + [0.05] * 3,
            "mixing_schedule": (1.0, 0, 3000),
            "torque_supervision": False,
            "torque_supervision_schedule": (0.0, 1000, 1000),
            "dagger_update_freq": 20,
            "priv_reg_schedule": (0.0, 0.1, 3000, 7000),
        },
        "runner": {"num_steps_per_env": 40, "save_interval": 500},
    }

