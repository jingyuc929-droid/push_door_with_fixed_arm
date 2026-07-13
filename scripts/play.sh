python scripts/rsl_rl/play.py \
    --task Template-Door-Env-v0 \
    --num_envs 1  \
    --checkpoint logs/rsl_rl/door_asymmetric_critic/2026-07-13_11-55-52_privileged_latent_teacher_distill_ready_v1/model_1400.pt    \
    --agent rsl_rl_teacher_cfg_entry_point