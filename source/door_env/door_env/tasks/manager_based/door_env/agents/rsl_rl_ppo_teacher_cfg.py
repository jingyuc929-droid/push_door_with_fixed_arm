from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DoorEnvPPOTeacherCfg(RslRlOnPolicyRunnerCfg):
    # ---- runner ----
    seed = 42
    device = "cuda:0"

    # 每次 rollout 的步数（先用小一点更稳，后面再调大）
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 10

    experiment_name = "door_privileged_teacher"
    run_name = "clean_obs_stage_gated_v1"

    # 关键：把环境的 observation groups 映射到算法使用的集合
    # teacher 训练：actor/critic 都吃 policy + teacher
    obs_groups = {
        "policy": ["policy", "teacher"],
        "critic": ["policy", "teacher"],
    }

    # 可选：动作裁剪（如果你看到 policy 输出很大再开）
    clip_actions = 1.0

    # ---- policy net ----
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )

    # ---- PPO ----
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
