from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class DoorBotPrivilegedActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "DoorBotPrivilegedActorCritic"
    z_priv_dim = 16
    privileged_encoder_hidden_dims = [64, 32]
    clean_obs_group = "policy_obs_clean"
    privileged_obs_group = "privileged_state"
    privileged_obs_normalization = True

@configclass
class DoorEnvPPOTeacherCfg(RslRlOnPolicyRunnerCfg):
    # ---- runner ----
    seed = 42
    device = "cuda:0"

    # 每次 rollout 的步数（先用小一点更稳，后面再调大）
    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 50

    experiment_name = "door_asymmetric_critic"
    # Teacher PPO first; Student-RNN optimization will use a separate future
    # distillation config while consuming the rollout interface from this run.
    run_name = "privileged_latent_teacher_distill_ready_v1"
    class_name = "DoorBotTeacherRunner"
    history_length = 32
    # Teacher PPO 默认不保存 Student 蒸馏 rollout；需要采集数据时显式开启。
    collect_distillation_rollout = False
    stage_transition_thresholds = {
        "initial_push": 0.02,
        "push_follow": 0.10,
        "traverse": 0.70,
    }

    # Teacher actor internally encodes privileged_state to z_priv and consumes
    # [policy_obs_clean, z_priv]. The critic consumes clean + privileged state.
    obs_groups = {
        "policy": ["policy_obs_clean"],
        "critic": ["policy_obs_clean", "privileged_state"],
    }

    # 可选：动作裁剪（如果你看到 policy 输出很大再开）
    clip_actions = 1.0

    # ---- policy net ----
    policy = DoorBotPrivilegedActorCriticCfg(
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
