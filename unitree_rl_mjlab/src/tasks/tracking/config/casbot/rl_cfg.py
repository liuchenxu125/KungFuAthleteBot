"""RL configuration for Casbot Skeleton tracking task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

from src.rl.config import HolosomaFastSACRunnerCfg


def casbot_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for Casbot tracking task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="casbot_tracking",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30001,
    )


def casbot_tracking_fastsac_runner_cfg() -> HolosomaFastSACRunnerCfg:
    """FastSAC config for Casbot motion tracking."""
    return HolosomaFastSACRunnerCfg(
        experiment_name="casbot_tracking_fastsac",
        max_iterations=720024,
        save_interval=500,
        logging_interval=100,
        console_logging_interval=100,
        gamma=0.99,
        num_updates=2,
        num_atoms=501,
        v_min=-20.0,
        v_max=20.0,
        target_entropy_ratio=0.5,
        policy_frequency=4,
        batch_size=2048,
        buffer_size=4096,
        use_symmetry=False,
        use_tanh=False,
        compile=False,
        amp=False,
    )
