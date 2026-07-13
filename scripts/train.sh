#!/usr/bin/env bash
set -euo pipefail

# Piper hook door-opening teacher training.
# Override defaults from the shell, for example:


TASK="${TASK:-Template-Door-Env-v0}"
AGENT="${AGENT:-rsl_rl_teacher_cfg_entry_point}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-10000}"
RUN_NAME="${RUN_NAME:-privileged_latent_teacher_distill_ready_v1}"
CONSOLE_LOG_MODE="${DOORBOT_CONSOLE_LOG_MODE:-stage_and_termination}"
PYTHON_BIN="${PYTHON_BIN:-/home/jing/anaconda3/envs/isaac/bin/python}"



export DOORBOT_CONSOLE_LOG_MODE="${CONSOLE_LOG_MODE}"

exec "${PYTHON_BIN}" scripts/rsl_rl/train.py \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --agent "${AGENT}" \
    --run_name "${RUN_NAME}" \
    --headless  
    "$@"
