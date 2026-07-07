"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append Teleop cli args
parser.add_argument("--teleop", action="store_true", default=False, help="Enable SpaceMouse teleoperation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import builtins
import logging
import os
import time
import torch
import types
from datetime import datetime

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import door_env.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


_CONSOLE_METRIC_ORDER = (
    "Episode_Reward/stage_gated_door_reward",
    "Episode_Termination/time_out",
    "Episode_Termination/door_open_success",
    "Episode_Termination/release_after_grasp_failure",
    "Episode_Termination/release_after_unlock_failure",
    "stage_gated_reward/align",
    "stage_gated_reward/approach",
    "stage_gated_reward/close_when_ready",
    "stage_gated_reward/grasp_handle",
    "stage_gated_reward/grasp_quality_keep",
    "stage_gated_reward/grasp_quality_penalty",
    "stage_gated_reward/press_handle_raw",
    "stage_gated_reward/press_handle_gated",
    "stage_gated_reward/keep_handle_after_press",
    "stage_gated_reward/stall_after_grasp",
    "stage_gated_reward/stall_after_press",
    "stage_gated_reward/unlock_progress_raw",
    "stage_gated_reward/unlock_progress_gated",
    "stage_gated_reward/unlock_transition",
    "stage_gated_reward/push_door",
    "stage_gated_reward/stage0_reward",
    "stage_gated_reward/stage1_reward",
    "stage_gated_reward/stage2_reward",
    "stage_gated_reward/stage0_mask_ratio",
    "stage_gated_reward/stage1_mask_ratio",
    "stage_gated_reward/stage2_mask_ratio",
)
_CONSOLE_METRIC_PRIORITY = {name: idx for idx, name in enumerate(_CONSOLE_METRIC_ORDER)}


def _console_metric_name_from_line(line: str) -> str | None:
    stripped = line.strip()
    if "/" not in stripped or ":" not in stripped:
        return None
    name = stripped.split(":", 1)[0].strip()
    return name if "/" in name else None


def _should_print_console_metric(metric_name: str) -> bool:
    exact_names = {
        "Episode_Reward/stage_gated_door_reward",
    }
    allowed_prefixes = (
        "Episode_Termination/",
        "stage_gated_reward/",
        "termination/",
    )
    return metric_name in exact_names or metric_name.startswith(allowed_prefixes)


def _console_metric_sort_key(item: tuple[str, str]) -> tuple[int, str]:
    name, _line = item
    return (_CONSOLE_METRIC_PRIORITY.get(name, len(_CONSOLE_METRIC_PRIORITY)), name)


def _filter_console_log_text(text: str, mode: str) -> str:
    if mode == "all":
        return text

    lines = text.splitlines()
    kept_lines: list[str] = []
    metric_lines: dict[str, str] = {}
    insert_at = None

    for line in lines:
        metric_name = _console_metric_name_from_line(line)
        if metric_name is None:
            if insert_at is None and line.strip().startswith("Total timesteps:"):
                insert_at = len(kept_lines)
            kept_lines.append(line)
            continue

        if mode == "stage_and_termination" and _should_print_console_metric(metric_name):
            metric_lines.setdefault(metric_name, line)

    if mode == "minimal" or len(metric_lines) == 0:
        return "\n".join(kept_lines) + ("\n" if text.endswith("\n") else "")

    ordered_metric_lines = [line for _name, line in sorted(metric_lines.items(), key=_console_metric_sort_key)]
    if insert_at is None:
        insert_at = len(kept_lines)
    if insert_at > 0 and kept_lines[insert_at - 1].strip() != "":
        ordered_metric_lines.insert(0, "")
    if insert_at < len(kept_lines) and kept_lines[insert_at].strip() != "":
        ordered_metric_lines.append("")
    output_lines = kept_lines[:insert_at] + ordered_metric_lines + kept_lines[insert_at:]
    return "\n".join(output_lines) + ("\n" if text.endswith("\n") else "")


def _patch_runner_console_log(runner, mode: str):
    if mode == "all" or not hasattr(runner, "log"):
        return runner

    original_log = runner.log

    def filtered_log(self, *args, **kwargs):
        original_print = builtins.print
        original_logger_info = logging.Logger.info

        def filtered_print(*print_args, **print_kwargs):
            if len(print_args) == 1 and isinstance(print_args[0], str):
                print_args = (_filter_console_log_text(print_args[0], mode),)
            return original_print(*print_args, **print_kwargs)

        def filtered_logger_info(logger_self, msg, *info_args, **info_kwargs):
            if isinstance(msg, str):
                msg = _filter_console_log_text(msg, mode)
            return original_logger_info(logger_self, msg, *info_args, **info_kwargs)

        builtins.print = filtered_print
        logging.Logger.info = filtered_logger_info
        try:
            return original_log(*args, **kwargs)
        finally:
            builtins.print = original_print
            logging.Logger.info = original_logger_info

    runner.log = types.MethodType(filtered_log, runner)
    return runner


def _resolve_resume_mode(args_cli, agent_cfg) -> str:
    resume_mode = getattr(args_cli, "resume_mode", None)
    if resume_mode is None:
        return "full" if agent_cfg.resume else "none"
    if resume_mode in ("full", "finetune") and not agent_cfg.resume:
        raise ValueError(f"--resume_mode={resume_mode} requires --resume.")
    if resume_mode == "none" and agent_cfg.resume:
        raise ValueError("--resume and --resume_mode=none are contradictory. Drop --resume or use full/finetune.")
    return resume_mode


def _load_finetune_checkpoint(runner, resume_path: str, device: str):
    try:
        infos = runner.load(resume_path, load_optimizer=False)
    except TypeError:
        checkpoint = torch.load(resume_path, weights_only=False, map_location=device)
        policy = getattr(runner.alg, "policy", None)
        if policy is None:
            policy = getattr(runner.alg, "actor_critic", None)
        if policy is None:
            raise RuntimeError("Could not find runner.alg.policy or runner.alg.actor_critic for finetune loading.")
        policy.load_state_dict(checkpoint["model_state_dict"])
        infos = checkpoint.get("infos", {})

    runner.current_learning_iteration = 0
    return infos


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    resume_mode = _resolve_resume_mode(args_cli, agent_cfg)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    should_load_checkpoint = resume_mode != "none" or agent_cfg.algorithm.class_name == "Distillation"
    if should_load_checkpoint:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap for teleoperation
    if args_cli.teleop:
        from door_env.wrappers.teleop_wrapper import TeleopWrapper
        print("[INFO] Enabling SpaceMouse teleoperation wrapper.")
        env = TeleopWrapper(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------------------
    # GUI HUD: handle/door joint state + door_unlocked + contact sensor forces
    # No terminal prints. Works only in GUI mode (headless=False).
    # ---------------------------------------------------------------------
    _hud_win, _hud_sub = None, None
    if not args_cli.headless:
        try:
            import omni.ui as ui
            import omni.kit.app
    
            from isaacsim.util.debug_draw import _debug_draw

            base_env = env.unwrapped
            door = base_env.scene["door"]
            robot = base_env.scene["robot"]

            draw = _debug_draw.acquire_debug_draw_interface()

            # ---- debug visualization params (match cfg you want to inspect) ----
            VIS_EE_OFFSET = (0.1523, 0.0, 0.0)
            VIS_HANDLE_OFFSET_H = (-0.08, 0.04, 0.01)

            VIS_GRIPPER_OPEN_AXIS_HAND = (0.0, 1.0, 0.0)
            VIS_GRIPPER_APPROACH_AXIS_HAND = (1.0, 0.0, 0.0)
            VIS_HANDLE_APPROACH_AXIS = 1   # handle local +y

            VIS_AXIS_LEN = 0.08
            VIS_POINT_SIZE = 14.0
            VIS_LINE_WIDTH = 3.0

            # try to get filtered contact sensors from the scene
            try:
                left_sensor = base_env.scene["left_finger_contact"]
            except Exception:
                left_sensor = None
            try:
                right_sensor = base_env.scene["right_finger_contact"]
            except Exception:
                right_sensor = None

            def _filtered_force_norm(sensor):
                """Sum of norms of latest filtered contact force vectors for each env."""
                if sensor is None:
                    return None
                try:
                    fm = sensor.data.force_matrix_w
                    if fm is None:
                        return None
                    if fm.numel() == 0:
                        return torch.zeros(base_env.num_envs, device=base_env.device)
                    fm = fm.reshape(fm.shape[0], -1, 3)
                    return torch.linalg.norm(fm, dim=-1).sum(dim=-1)
                except Exception:
                    return None

        # -----------------------------
        # tiny quaternion/vector helpers
        # -----------------------------
            def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
            # q: [...,4] in wxyz
                w, x, y, z = q.unbind(-1)
                return torch.stack((w, -x, -y, -z), dim=-1)

            def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
                w1, x1, y1, z1 = q1.unbind(-1)
                w2, x2, y2, z2 = q2.unbind(-1)
                w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
                x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
                y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
                z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
                return torch.stack((w, x, y, z), dim=-1)

            def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # q: [N,4], v: [N,3]
                qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
                return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]

            def _safe_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
                return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)

            def _local_vec_to_world(q: torch.Tensor, v_local: tuple[float, float, float]) -> torch.Tensor:
                v = torch.tensor(v_local, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
                return _quat_rotate(q, v)

            def _axis_idx_to_local_vec(axis_idx: int, device, dtype) -> torch.Tensor:
                e = torch.zeros((1, 3), device=device, dtype=dtype)
                e[0, int(axis_idx)] = 1.0
                return e

            def _handle_axis_world(qH: torch.Tensor, axis_idx: int) -> torch.Tensor:
                e = _axis_idx_to_local_vec(axis_idx, qH.device, qH.dtype).repeat(qH.shape[0], 1)
                return _quat_rotate(qH, e)

            def _body_pose_w(asset, body_id: int, env_idx: int):
                pos = asset.data.body_pos_w[env_idx, body_id].detach()
                quat = asset.data.body_quat_w[env_idx, body_id].detach()
                return pos, quat

            def _draw_axis(draw, origin, direction, color, length=0.08, width=3.0):
                p0 = tuple(origin.tolist())
                p1 = tuple((origin + direction * length).tolist())
                draw.draw_lines([p0], [p1], [color], [width])
        # -----------------------------
        # resolve ids once
        # -----------------------------
            joint_names = list(door.data.joint_names)

            if "handle_joint" in joint_names:
                handle_jid = joint_names.index("handle_joint")
                handle_name = "handle_joint"
            else:
                cand = [i for i, n in enumerate(joint_names) if "handle" in n.lower()]
                handle_jid = cand[0] if cand else 0
                handle_name = joint_names[handle_jid]

            if "door_joint" in joint_names:
                door_jid = joint_names.index("door_joint")
                door_name = "door_joint"
            else:
                cand = [i for i, n in enumerate(joint_names) if "door" in n.lower()]
                door_jid = cand[0] if cand else None
                door_name = joint_names[door_jid] if door_jid is not None else "N/A"

            robot_body_names = list(robot.data.body_names)
            door_body_names = list(door.data.body_names)

            def _find_body_id(names, target):
                if target in names:
                    return names.index(target)
                cand = [i for i, n in enumerate(names) if target.lower() in n.lower()]
                if not cand:
                    raise RuntimeError(f"Body '{target}' not found in {names}")
                return cand[0]

            left_bid = _find_body_id(robot_body_names, "link7")
            right_bid = _find_body_id(robot_body_names, "link8")
            hand_bid = _find_body_id(robot_body_names, "link6")
            handle_bid = _find_body_id(door_body_names, "handle_1")

        # -----------------------------
        # align-v2 params (match cfg)
        # -----------------------------
            grasp_axis = 2
            min_sep = 0.010
            sep_scale = 0.010
            symmetry_scale = 0.015
            gripper_open_axis_hand = (0.0, 1.0, 0.0)
            gripper_approach_axis_hand = (1.0, 0.0, 0.0)
            handle_approach_axis = 1
            align_side_weight = 0.70
            align_open_weight = 0.10
            align_approach_weight = 0.20
            align_threshold = 0.30

            gs_distance_threshold = 0.10
            gs_force_threshold = 1.0
            gs_relax_near_after_handle_pos = -0.05
            gs_open_width = 0.08
            gs_min_closedness = 0.5
            gs_require_wrap = True
            gs_require_any_finger_contact = False
            gq_near_sigma = 0.06
            gq_near_hard_threshold = 0.14
            gq_expected_approach_sign = 1.0
            gq_contact_threshold = 0.25
            gq_contact_scale = 0.50
            gq_balance_power = 0.5
            gq_open_width = 0.09
            gq_min_closedness = 0.35
            gq_target_closedness = 0.65
            gq_max_closedness = 0.98
            gq_single_force_high = 1.0
            gq_single_force_low = 0.15
            gq_quality_gate_floor = 0.15
            
            robot_joint_names = list(robot.data.joint_names)

            gripper_jids = [i for i, n in enumerate(robot_joint_names) if ("gripper" in n.lower()) or ("finger" in n.lower())]

            if len(gripper_jids) == 0:
                gripper_jids = [robot_joint_names.index("gripper_joint")]

            HUD_ENV_INDEX = 276

        # HUD window
            _hud_win = ui.Window("Door Train HUD", width=520, height=520)
            with _hud_win.frame:
                with ui.VStack(spacing=4):
                    lbl_handle_pos = ui.Label(f"{handle_name} pos (env{HUD_ENV_INDEX}): --")
                    lbl_handle_vel = ui.Label(f"{handle_name} vel (env{HUD_ENV_INDEX}): --")
                    lbl_handle_prog = ui.Label(f"handle progress (env{HUD_ENV_INDEX}): --")
                    lbl_door_pos = ui.Label(f"{door_name} pos (env{HUD_ENV_INDEX}): --")
                    lbl_door_vel = ui.Label(f"{door_name} vel (env{HUD_ENV_INDEX}): --")
                    lbl_unlock = ui.Label(f"door_unlocked (env{HUD_ENV_INDEX}): --")
                    lbl_force_l = ui.Label(f"left sensor |F| (env{HUD_ENV_INDEX}): --")
                    lbl_force_r = ui.Label(f"right sensor |F| (env{HUD_ENV_INDEX}): --")
                    lbl_grip_width = ui.Label(f"gripper width(raw, env{HUD_ENV_INDEX}): --")
                    lbl_approach_dist = ui.Label(f"approach tcp->grasp dist (env{HUD_ENV_INDEX}): --")
                    
                    ui.Separator()
                    lbl_gs_gate = ui.Label(f"grasp_success gate (env{HUD_ENV_INDEX}): --")
                    lbl_gs_archive = ui.Label(f"grasp_start archive: --")
                    lbl_gs_mode = ui.Label(f"strict / relaxed / final (env{HUD_ENV_INDEX}): -- / -- / --")
                    lbl_gs_near = ui.Label(f"near_ok (env{HUD_ENV_INDEX}): --")
                    lbl_gs_wrap = ui.Label(f"wrap_ok (env{HUD_ENV_INDEX}): --")
                    lbl_gs_contact = ui.Label(f"contact_ok (env{HUD_ENV_INDEX}): --")
                    lbl_gs_close = ui.Label(f"close_ok (env{HUD_ENV_INDEX}): --")
                    lbl_gs_relax = ui.Label(f"relax_phase (env{HUD_ENV_INDEX}): --")
                    lbl_gs_dist = ui.Label(f"dist_to_grasp (env{HUD_ENV_INDEX}): --")
                    lbl_gs_wrap_score = ui.Label(f"wrap_score (env{HUD_ENV_INDEX}): --")
                    lbl_gs_force = ui.Label(f"fL / fR (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gs_closeval = ui.Label(f"width / closedness (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gs_hold = ui.Label(f"hold_counter / given (env{HUD_ENV_INDEX}): -- / --")

                    ui.Separator()
                    lbl_gq_quality = ui.Label(f"grasp_quality / gate (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gq_dist = ui.Label(f"tcp_to_grasp_dist / finger_mid_to_grasp_dist (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gq_near_wrap = ui.Label(f"near_score / wrap_score (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gq_pose = ui.Label(f"open_align / approach_dot_raw / approach_align (env{HUD_ENV_INDEX}): -- / -- / --")
                    lbl_gq_contact = ui.Label(f"contact_score / contact_ok (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gq_balance = ui.Label(f"balance_score / closure_score (env{HUD_ENV_INDEX}): -- / --")
                    lbl_gq_bad = ui.Label(f"closed_no_contact / single_finger (env{HUD_ENV_INDEX}): -- / --")

                    ui.Separator()
                    lbl_press_gate = ui.Label(f"press gate (env{HUD_ENV_INDEX}): --")
                    lbl_press_phase = ui.Label(f"press phase / contact / close (env{HUD_ENV_INDEX}): -- / -- / --")
                    lbl_press_prog = ui.Label(f"press dpos / prog_gate (env{HUD_ENV_INDEX}): -- / --")
                    lbl_press_vel = ui.Label(f"press vel_raw / vel_ema / desired (env{HUD_ENV_INDEX}): -- / -- / --")
                    lbl_press_shaping = ui.Label(f"press r_pos / r_neg / r (env{HUD_ENV_INDEX}): -- / -- / --")
                    lbl_stall = ui.Label(f"stall bad / depth (env{HUD_ENV_INDEX}): -- / --")

        # 20 Hz refresh is enough and lightweight
            _hud_dt = 1.0 / 20.0
            _hud_last_t = 0.0
            _handle_start = 0.0
            _handle_threshold = -0.3
            _handle_denom = (_handle_threshold - _handle_start) if abs(_handle_threshold - _handle_start) > 1e-9 else None

            def _on_hud_update(_evt):
                nonlocal _hud_last_t
                now = time.time()
                if now - _hud_last_t < _hud_dt:
                    return
                _hud_last_t = now

                env_index = HUD_ENV_INDEX
                if env_index >= base_env.num_envs:
                    lbl_handle_pos.text = (
                        f"{handle_name} pos (env{HUD_ENV_INDEX}): N/A "
                        f"(num_envs={base_env.num_envs})"
                    )
                    lbl_handle_vel.text = f"{handle_name} vel (env{HUD_ENV_INDEX}): N/A"
                    lbl_handle_prog.text = f"handle progress (env{HUD_ENV_INDEX}): N/A"
                    lbl_door_pos.text = f"{door_name} pos (env{HUD_ENV_INDEX}): N/A"
                    lbl_door_vel.text = f"{door_name} vel (env{HUD_ENV_INDEX}): N/A"
                    lbl_unlock.text = f"door_unlocked (env{HUD_ENV_INDEX}): N/A"
                    lbl_force_l.text = f"left sensor |F| (env{HUD_ENV_INDEX}): N/A"
                    lbl_force_r.text = f"right sensor |F| (env{HUD_ENV_INDEX}): N/A"
                    lbl_grip_width.text = f"gripper width(raw, env{HUD_ENV_INDEX}): N/A"
                    lbl_approach_dist.text = f"approach tcp->grasp dist (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_gate.text = f"grasp_success gate (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_archive.text = "grasp_start archive: N/A"
                    lbl_gs_mode.text = f"strict / relaxed / final (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_gs_near.text = f"near_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_wrap.text = f"wrap_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_contact.text = f"contact_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_close.text = f"close_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_relax.text = f"relax_phase (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_dist.text = f"dist_to_grasp (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_wrap_score.text = f"wrap_score (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_force.text = f"fL / fR (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gs_closeval.text = f"width / closedness (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gs_hold.text = f"hold_counter / given (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_quality.text = f"grasp_quality / gate (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_dist.text = f"tcp_to_grasp_dist / finger_mid_to_grasp_dist (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_near_wrap.text = f"near_score / wrap_score (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_pose.text = f"open_align / approach_dot_raw / approach_align (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_gq_contact.text = f"contact_score / contact_ok (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_balance.text = f"balance_score / closure_score (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_bad.text = f"closed_no_contact / single_finger (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_press_gate.text = f"press gate (env{HUD_ENV_INDEX}): N/A"
                    lbl_press_phase.text = f"press phase / contact / close (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_press_prog.text = f"press dpos / prog_gate (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_press_vel.text = f"press vel_raw / vel_ema / desired (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_press_shaping.text = f"press r_pos / r_neg / r (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_stall.text = f"stall bad / depth (env{HUD_ENV_INDEX}): N/A / N/A"
                    return

            # -----------------------------
            # existing joint/contact HUD
            # -----------------------------
                try:
                    hpos = float(door.data.joint_pos[env_index, handle_jid].item())
                    hvel = float(door.data.joint_vel[env_index, handle_jid].item())
                    lbl_handle_pos.text = f"{handle_name} pos (env{HUD_ENV_INDEX}): {hpos:.6f}"
                    lbl_handle_vel.text = f"{handle_name} vel (env{HUD_ENV_INDEX}): {hvel:.6f}"

                    if _handle_denom is None:
                        hprog = 0.0
                    else:
                        hprog = (hpos - _handle_start) / _handle_denom
                        hprog = 0.0 if hprog < 0.0 else (1.0 if hprog > 1.0 else hprog)
                    lbl_handle_prog.text = (
                        f"handle progress (env{HUD_ENV_INDEX}): {hprog:.3f}   (0→{_handle_threshold})"
                    )
                except Exception:
                    pass

                if door_jid is not None:
                    try:
                        dpos = float(door.data.joint_pos[env_index, door_jid].item())
                        dvel = float(door.data.joint_vel[env_index, door_jid].item())
                        lbl_door_pos.text = f"{door_name} pos (env{HUD_ENV_INDEX}): {dpos:.6f}"
                        lbl_door_vel.text = f"{door_name} vel (env{HUD_ENV_INDEX}): {dvel:.6f}"
                    except Exception:
                        pass
                else:
                    lbl_door_pos.text = f"{door_name} pos (env{HUD_ENV_INDEX}): N/A"
                    lbl_door_vel.text = f"{door_name} vel (env{HUD_ENV_INDEX}): N/A"

                try:
                    if hasattr(base_env, "_door_unlocked"):
                        unlocked = bool(base_env._door_unlocked[env_index].item())
                        lbl_unlock.text = f"door_unlocked (env{HUD_ENV_INDEX}): {unlocked}"
                    else:
                        lbl_unlock.text = f"door_unlocked (env{HUD_ENV_INDEX}): N/A"
                except Exception:
                    lbl_unlock.text = f"door_unlocked (env{HUD_ENV_INDEX}): N/A"

                try:
                    fL = _filtered_force_norm(left_sensor)
                    if fL is None:
                        lbl_force_l.text = f"left sensor |F| (env{HUD_ENV_INDEX}): N/A"
                    else:
                        lbl_force_l.text = f"left sensor |F| (env{HUD_ENV_INDEX}): {float(fL[env_index].item()):.4f}"
                except Exception:
                    lbl_force_l.text = f"left sensor |F| (env{HUD_ENV_INDEX}): N/A"

                try:
                    fR = _filtered_force_norm(right_sensor)
                    if fR is None:
                        lbl_force_r.text = f"right sensor |F| (env{HUD_ENV_INDEX}): N/A"
                    else:
                        lbl_force_r.text = f"right sensor |F| (env{HUD_ENV_INDEX}): {float(fR[env_index].item()):.4f}"
                except Exception:
                    lbl_force_r.text = f"right sensor |F| (env{HUD_ENV_INDEX}): N/A"


                try:
                    # 和 reward 中 width 定义尽量一致：多关节就求和，单关节就直接取值
                    if len(gripper_jids) == 1:
                        width_raw = robot.data.joint_pos[env_index, gripper_jids[0]]
                    else:
                        width_raw = robot.data.joint_pos[env_index, gripper_jids].sum()


                    lbl_grip_width.text = f"gripper width(raw, env{HUD_ENV_INDEX}): {float(width_raw.item()):.5f}"
                   
                except Exception:
                    lbl_grip_width.text = f"gripper width(raw, env{HUD_ENV_INDEX}): N/A"
                    
                # -----------------------------
                # debug draw for ee_tcp / handle grasp point / axes
                # -----------------------------
                try:
                    draw.clear_points()
                    draw.clear_lines()

                    # current env index
                    env_i = env_index

                    # body poses
                    hand_pos, hand_quat = _body_pose_w(robot, hand_bid, env_i)
                    handle_pos, handle_quat = _body_pose_w(door, handle_bid, env_i)

                    # offsets in local frames
                    ee_off = torch.tensor(VIS_EE_OFFSET, device=hand_pos.device, dtype=hand_pos.dtype).unsqueeze(0)
                    h_off = torch.tensor(VIS_HANDLE_OFFSET_H, device=handle_pos.device, dtype=handle_pos.dtype).unsqueeze(0)

                    hand_quat_b = hand_quat.unsqueeze(0)
                    handle_quat_b = handle_quat.unsqueeze(0)

                    ee_tcp = hand_pos + _quat_rotate(hand_quat_b, ee_off)[0]
                    handle_grasp = handle_pos + _quat_rotate(handle_quat_b, h_off)[0]
                    approach_dist = torch.linalg.norm(ee_tcp - handle_grasp)

                    # local hand axes -> world
                    open_axis_local = torch.tensor(VIS_GRIPPER_OPEN_AXIS_HAND, device=hand_pos.device, dtype=hand_pos.dtype).unsqueeze(0)
                    approach_axis_local = torch.tensor(VIS_GRIPPER_APPROACH_AXIS_HAND, device=hand_pos.device, dtype=hand_pos.dtype).unsqueeze(0)

                    open_axis_w = _quat_rotate(hand_quat_b, open_axis_local)[0]
                    approach_axis_w = _quat_rotate(hand_quat_b, approach_axis_local)[0]

                    # handle local approach axis -> world
                    handle_axis_local = torch.zeros((1, 3), device=handle_pos.device, dtype=handle_pos.dtype)
                    handle_axis_local[0, VIS_HANDLE_APPROACH_AXIS] = 1.0
                    handle_axis_w = _quat_rotate(handle_quat_b, handle_axis_local)[0]

                    # points
                    pts = [
                        tuple(handle_pos.tolist()),      # handle origin
                        tuple(handle_grasp.tolist()),    # handle grasp point
                        tuple(hand_pos.tolist()),        # link6 origin
                        tuple(ee_tcp.tolist()),          # ee tcp
                    ]
                    pt_colors = [
                        (1.0, 1.0, 0.0, 1.0),   # yellow
                        (1.0, 0.0, 1.0, 1.0),   # magenta
                        (0.0, 1.0, 1.0, 1.0),   # cyan
                        (1.0, 1.0, 1.0, 1.0),   # white
                    ]
                    pt_sizes = [VIS_POINT_SIZE] * len(pts)
                    draw.draw_points(pts, pt_colors, pt_sizes)

                    # lines: offsets
                    line_starts = [
                        tuple(hand_pos.tolist()),
                        tuple(handle_pos.tolist()),
                        tuple(ee_tcp.tolist()),
                    ]
                    line_ends = [
                        tuple(ee_tcp.tolist()),
                        tuple(handle_grasp.tolist()),
                        tuple(handle_grasp.tolist()),
                    ]
                    line_colors = [
                        (0.7, 0.7, 0.7, 1.0),   # hand -> tcp
                        (1.0, 0.5, 0.0, 1.0),   # handle -> grasp point
                        (1.0, 1.0, 1.0, 1.0),   # approach distance used by reward
                    ]
                    line_widths = [VIS_LINE_WIDTH, VIS_LINE_WIDTH, VIS_LINE_WIDTH]
                    draw.draw_lines(line_starts, line_ends, line_colors, line_widths)
                    lbl_approach_dist.text = (
                        f"approach tcp->grasp dist (env{HUD_ENV_INDEX}): {float(approach_dist.item()):.4f}"
                    )

                    # axes from ee_tcp / handle grasp point
                    _draw_axis(draw, ee_tcp, open_axis_w, (0.0, 1.0, 0.0, 1.0), length=VIS_AXIS_LEN, width=VIS_LINE_WIDTH)       # green
                    _draw_axis(draw, ee_tcp, approach_axis_w, (1.0, 0.0, 0.0, 1.0), length=VIS_AXIS_LEN, width=VIS_LINE_WIDTH)   # red
                    _draw_axis(draw, handle_grasp, handle_axis_w, (0.0, 0.4, 1.0, 1.0), length=VIS_AXIS_LEN, width=VIS_LINE_WIDTH)  # blue

                except Exception as e:
                    # keep training running even if visualization fails
                    lbl_approach_dist.text = f"approach tcp->grasp dist (env{HUD_ENV_INDEX}): N/A"
                    pass

                # -----------------------------
                # grasp_success gate HUD (selected env only)
                # -----------------------------
                try:
                    archive_grasp = base_env.__dict__.get("_archive_grasp", None)
                    if archive_grasp is None:
                        lbl_gs_archive.text = "grasp_start archive: empty"
                    else:
                        archive_size = int(archive_grasp.get("size", 0))
                        archive_cap = int(archive_grasp.get("cap", 0))
                        archive_ptr = int(archive_grasp.get("ptr", 0))
                        lbl_gs_archive.text = (
                            f"grasp_start archive: size={archive_size} / {archive_cap}, ptr={archive_ptr}"
                        )

                    env_i = env_index

                    # body poses
                    pL = robot.data.body_pos_w[env_i:env_i+1, left_bid, :]
                    pR = robot.data.body_pos_w[env_i:env_i+1, right_bid, :]
                    qG = robot.data.body_quat_w[env_i:env_i+1, hand_bid, :]
                    pH = door.data.body_pos_w[env_i:env_i+1, handle_bid, :]
                    qH = door.data.body_quat_w[env_i:env_i+1, handle_bid, :]

                    # --- near_ok: fingertip-mid to handle grasp point ---
                    p_mid = 0.5 * (pL + pR)
                    h_off = torch.tensor(VIS_HANDLE_OFFSET_H, device=pH.device, dtype=pH.dtype).unsqueeze(0)
                    p_grasp = pH + _quat_rotate(qH, h_off)
                    dist = torch.linalg.norm(p_mid - p_grasp, dim=-1)
                    near_ok = dist < gs_distance_threshold

                    # --- wrap_ok: same geometry as reward, but only keep final wrap score ---
                    qHc = _quat_conjugate(qH)
                    L_h = _quat_rotate(qHc, pL - pH)
                    R_h = _quat_rotate(qHc, pR - pH)

                    l = L_h[:, grasp_axis]
                    r = R_h[:, grasp_axis]

                    opposite = (l * r < 0.0).float()
                    sep = torch.abs(l - r)
                    sep_score = torch.tanh(torch.clamp(sep - float(min_sep), min=0.0) / float(sep_scale))

                    sym_err = torch.abs(torch.abs(l) - torch.abs(r))
                    sym_score = torch.exp(-sym_err / float(symmetry_scale))

                    side_score = opposite * sep_score * sym_score

                    g_open_w = _safe_unit(_local_vec_to_world(qG, gripper_open_axis_hand))
                    h_grasp_w = _safe_unit(_handle_axis_world(qH, grasp_axis))
                    open_dot = torch.sum(g_open_w * h_grasp_w, dim=-1)
                    open_align = torch.abs(open_dot).clamp(0.0, 1.0)

                    g_app_w = _safe_unit(_local_vec_to_world(qG, gripper_approach_axis_hand))
                    h_app_w = _safe_unit(_handle_axis_world(qH, handle_approach_axis))
                    app_dot = torch.sum(g_app_w * h_app_w, dim=-1)
                    approach_align = torch.abs(app_dot).clamp(0.0, 1.0)

                    score_open = side_score * open_align
                    score_app = side_score * approach_align
                    wsum = float(align_side_weight + align_open_weight + align_approach_weight)
                    wrap_score = (
                        float(align_side_weight) * side_score
                        + float(align_open_weight) * score_open
                        + float(align_approach_weight) * score_app
                    ) / max(wsum, 1e-6)
                    wrap_score = wrap_score.clamp(0.0, 1.0)

                    if gs_require_wrap:
                        wrap_ok = wrap_score >= align_threshold
                    else:
                        wrap_ok = torch.ones_like(near_ok, dtype=torch.bool)

                    # --- contact_ok ---
                    fL_all = _filtered_force_norm(left_sensor)
                    fR_all = _filtered_force_norm(right_sensor)
                    if fL_all is None or fR_all is None:
                        fL = torch.tensor(float("nan"), device=base_env.device)
                        fR = torch.tensor(float("nan"), device=base_env.device)
                        contact_ok = torch.tensor(False, device=base_env.device)
                    else:
                        fL = fL_all[env_i]
                        fR = fR_all[env_i]
                        if gs_require_any_finger_contact:
                            contact_ok = torch.maximum(fL, fR) > gs_force_threshold
                        else:
                            contact_ok = torch.minimum(fL, fR) > gs_force_threshold

                    # --- close_ok ---
                    if len(gripper_jids) == 1:
                        width = robot.data.joint_pos[env_i, gripper_jids[0]]
                    else:
                        width = robot.data.joint_pos[env_i, gripper_jids].sum()

                    closedness = 1.0 - torch.clamp(width / gs_open_width, 0.0, 1.0)
                    close_ok = closedness > gs_min_closedness

                    # --- relax phase + final gate ---
                    handle_pos_cur = door.data.joint_pos[env_i, handle_jid]
                    relax_phase = handle_pos_cur < gs_relax_near_after_handle_pos

                    gate_strict = near_ok[0] & wrap_ok[0] & contact_ok & close_ok
                    gate_relaxed = contact_ok & close_ok
                    gate_final = gate_relaxed if bool(relax_phase.item()) else gate_strict

                    # --- bonus internal state if available ---
                    hold_counter = "N/A"
                    given_flag = "N/A"
                    if hasattr(base_env, "_grasp_success_counter"):
                        hold_counter = str(int(base_env._grasp_success_counter[env_i].item()))
                    if hasattr(base_env, "_grasp_success_given"):
                        given_flag = str(bool(base_env._grasp_success_given[env_i].item()))

                    # --- HUD text ---
                    lbl_gs_gate.text = f"grasp_success gate (env{HUD_ENV_INDEX}): {bool(gate_final.item())}"
                    lbl_gs_mode.text = (
                        f"strict / relaxed / final (env{HUD_ENV_INDEX}): "
                        f"{bool(gate_strict.item())} / {bool(gate_relaxed.item())} / {bool(gate_final.item())}"
                    )
                    lbl_gs_near.text = f"near_ok (env{HUD_ENV_INDEX}): {bool(near_ok[0].item())}"
                    lbl_gs_wrap.text = f"wrap_ok (env{HUD_ENV_INDEX}): {bool(wrap_ok[0].item())}"
                    lbl_gs_contact.text = f"contact_ok (env{HUD_ENV_INDEX}): {bool(contact_ok.item())}"
                    lbl_gs_close.text = f"close_ok (env{HUD_ENV_INDEX}): {bool(close_ok.item())}"
                    lbl_gs_relax.text = f"relax_phase (env{HUD_ENV_INDEX}): {bool(relax_phase.item())}"
                    lbl_gs_dist.text = f"dist_to_grasp (env{HUD_ENV_INDEX}): {float(dist[0].item()):.4f}"
                    lbl_gs_wrap_score.text = f"wrap_score (env{HUD_ENV_INDEX}): {float(wrap_score[0].item()):.3f}"
                    lbl_gs_force.text = f"fL / fR (env{HUD_ENV_INDEX}): {float(fL.item()):.4f} / {float(fR.item()):.4f}"
                    lbl_gs_closeval.text = (
                        f"width / closedness (env{HUD_ENV_INDEX}): {float(width.item()):.4f} / {float(closedness.item()):.3f}"
                    )
                    lbl_gs_hold.text = f"hold_counter / given (env{HUD_ENV_INDEX}): {hold_counter} / {given_flag}"

                except Exception:
                    lbl_gs_gate.text = f"grasp_success gate (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_archive.text = "grasp_start archive: N/A"
                    lbl_gs_mode.text = f"strict / relaxed / final (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_gs_near.text = f"near_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_wrap.text = f"wrap_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_contact.text = f"contact_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_close.text = f"close_ok (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_relax.text = f"relax_phase (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_dist.text = f"dist_to_grasp (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_wrap_score.text = f"wrap_score (env{HUD_ENV_INDEX}): N/A"
                    lbl_gs_force.text = f"fL / fR (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gs_closeval.text = f"width / closedness (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gs_hold.text = f"hold_counter / given (env{HUD_ENV_INDEX}): N/A / N/A"

                # -----------------------------
                # Stage 1 grasp quality HUD (selected env only)
                # -----------------------------
                try:
                    env_i = env_index
                    pL = robot.data.body_pos_w[env_i:env_i+1, left_bid, :]
                    pR = robot.data.body_pos_w[env_i:env_i+1, right_bid, :]
                    pHand = robot.data.body_pos_w[env_i:env_i+1, hand_bid, :]
                    qHand = robot.data.body_quat_w[env_i:env_i+1, hand_bid, :]
                    pH = door.data.body_pos_w[env_i:env_i+1, handle_bid, :]
                    qH = door.data.body_quat_w[env_i:env_i+1, handle_bid, :]

                    ee_off = torch.tensor(VIS_EE_OFFSET, device=pHand.device, dtype=pHand.dtype).unsqueeze(0)
                    h_off = torch.tensor(VIS_HANDLE_OFFSET_H, device=pH.device, dtype=pH.dtype).unsqueeze(0)
                    p_tcp = pHand + _quat_rotate(qHand, ee_off)
                    p_target = pH + _quat_rotate(qH, h_off)
                    tcp_dist = torch.linalg.norm(p_tcp - p_target, dim=-1)
                    p_mid = 0.5 * (pL + pR)
                    finger_mid_dist = torch.linalg.norm(p_mid - p_target, dim=-1)

                    near_score = torch.exp(-torch.square(tcp_dist / float(gq_near_sigma)))
                    near_score = torch.where(
                        tcp_dist <= float(gq_near_hard_threshold),
                        near_score,
                        torch.zeros_like(near_score),
                    )

                    qHc = _quat_conjugate(qH)
                    L_h = _quat_rotate(qHc, pL - pH)
                    R_h = _quat_rotate(qHc, pR - pH)
                    l = L_h[:, grasp_axis]
                    r = R_h[:, grasp_axis]
                    opposite = (l * r < 0.0).float()
                    sep = torch.abs(l - r)
                    sep_score = torch.tanh(torch.clamp(sep - float(0.005), min=0.0) / float(sep_scale))
                    sym_err = torch.abs(torch.abs(l) - torch.abs(r))
                    sym_score = torch.exp(-sym_err / float(symmetry_scale))
                    gq_wrap_score = (opposite * sep_score * sym_score).clamp(0.0, 1.0)

                    g_open_w = _safe_unit(_local_vec_to_world(qHand, gripper_open_axis_hand))
                    h_grasp_w = _safe_unit(_handle_axis_world(qH, grasp_axis))
                    gq_open_align = torch.abs(torch.sum(g_open_w * h_grasp_w, dim=-1)).clamp(0.0, 1.0)

                    g_app_w = _safe_unit(_local_vec_to_world(qHand, gripper_approach_axis_hand))
                    h_app_w = _safe_unit(_handle_axis_world(qH, handle_approach_axis))
                    approach_dot_raw = torch.sum(g_app_w * h_app_w, dim=-1)
                    gq_approach_align = (float(gq_expected_approach_sign) * approach_dot_raw).clamp(0.0, 1.0)
                    pose_score = (0.40 * gq_wrap_score + 0.20 * gq_open_align + 0.40 * gq_approach_align).clamp(0.0, 1.0)

                    fL_all = _filtered_force_norm(left_sensor)
                    fR_all = _filtered_force_norm(right_sensor)
                    if fL_all is None or fR_all is None:
                        fL_q = torch.tensor(0.0, device=base_env.device)
                        fR_q = torch.tensor(0.0, device=base_env.device)
                    else:
                        fL_q = fL_all[env_i]
                        fR_q = fR_all[env_i]
                    f_min = torch.minimum(fL_q, fR_q)
                    f_max = torch.maximum(fL_q, fR_q)
                    gq_contact_ok = f_min > float(gq_contact_threshold)
                    contact_score = torch.tanh(
                        torch.clamp(f_min - float(gq_contact_threshold), min=0.0) / float(gq_contact_scale)
                    )
                    balance_score = torch.clamp(f_min / (f_max + 1.0e-6), 0.0, 1.0) ** float(gq_balance_power)

                    if len(gripper_jids) == 1:
                        width_q = 2.0 * robot.data.joint_pos[env_i, gripper_jids[0]]
                    else:
                        width_q = robot.data.joint_pos[env_i, gripper_jids].sum()
                    gq_closedness = 1.0 - torch.clamp(width_q / float(gq_open_width), 0.0, 1.0)
                    close_low = torch.clamp(
                        (gq_closedness - float(gq_min_closedness))
                        / float(max(gq_target_closedness - gq_min_closedness, 1e-6)),
                        0.0,
                        1.0,
                    )
                    close_high = torch.clamp(
                        (float(gq_max_closedness) - gq_closedness)
                        / float(max(gq_max_closedness - gq_target_closedness, 1e-6)),
                        0.0,
                        1.0,
                    )
                    closure_score = close_low * close_high

                    geometry_score = (
                        0.25 * near_score[0]
                        + 0.35 * gq_wrap_score[0]
                        + 0.25 * pose_score[0]
                        + 0.15 * closure_score
                    )
                    contact_quality = contact_score * (0.5 + 0.5 * balance_score)
                    grasp_quality = torch.clamp(geometry_score * contact_quality, 0.0, 1.0)
                    quality_gate = torch.clamp(
                        float(gq_quality_gate_floor) + (1.0 - float(gq_quality_gate_floor)) * grasp_quality,
                        0.0,
                        1.0,
                    )
                    closed_no_contact = (gq_closedness > 0.75) and (not bool(gq_contact_ok.item()))
                    single_finger = bool((f_max > float(gq_single_force_high)).item()) and bool(
                        (f_min < float(gq_single_force_low)).item()
                    )

                    lbl_gq_quality.text = (
                        f"grasp_quality / gate (env{HUD_ENV_INDEX}): "
                        f"{float(grasp_quality.item()):.3f} / {float(quality_gate.item()):.3f}"
                    )
                    lbl_gq_dist.text = (
                        f"tcp_to_grasp_dist / finger_mid_to_grasp_dist (env{HUD_ENV_INDEX}): "
                        f"{float(tcp_dist[0].item()):.4f} / {float(finger_mid_dist[0].item()):.4f}"
                    )
                    lbl_gq_near_wrap.text = (
                        f"near_score / wrap_score (env{HUD_ENV_INDEX}): "
                        f"{float(near_score[0].item()):.3f} / {float(gq_wrap_score[0].item()):.3f}"
                    )
                    lbl_gq_pose.text = (
                        f"open_align / approach_dot_raw / approach_align (env{HUD_ENV_INDEX}): "
                        f"{float(gq_open_align[0].item()):.3f} / {float(approach_dot_raw[0].item()):.3f} / "
                        f"{float(gq_approach_align[0].item()):.3f}"
                    )
                    lbl_gq_contact.text = (
                        f"contact_score / contact_ok (env{HUD_ENV_INDEX}): "
                        f"{float(contact_score.item()):.3f} / {bool(gq_contact_ok.item())}"
                    )
                    lbl_gq_balance.text = (
                        f"balance_score / closure_score (env{HUD_ENV_INDEX}): "
                        f"{float(balance_score.item()):.3f} / {float(closure_score.item()):.3f}"
                    )
                    lbl_gq_bad.text = (
                        f"closed_no_contact / single_finger (env{HUD_ENV_INDEX}): "
                        f"{closed_no_contact} / {single_finger}"
                    )

                except Exception:
                    lbl_gq_quality.text = f"grasp_quality / gate (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_dist.text = f"tcp_to_grasp_dist / finger_mid_to_grasp_dist (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_near_wrap.text = f"near_score / wrap_score (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_pose.text = (
                        f"open_align / approach_dot_raw / approach_align (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    )
                    lbl_gq_contact.text = f"contact_score / contact_ok (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_balance.text = f"balance_score / closure_score (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_gq_bad.text = f"closed_no_contact / single_finger (env{HUD_ENV_INDEX}): N/A / N/A"

                # -----------------------------
                # press gate HUD (selected env only)
                # -----------------------------
                try:
                    env_i = env_index

                    # shared filtered force norms
                    fL_all = _filtered_force_norm(left_sensor)
                    fR_all = _filtered_force_norm(right_sensor)
                    if fL_all is None or fR_all is None:
                        fL_cur = torch.tensor(float("nan"), device=base_env.device)
                        fR_cur = torch.tensor(float("nan"), device=base_env.device)
                        press_contact_ok = torch.tensor(False, device=base_env.device)
                        stall_contact_ok = torch.tensor(False, device=base_env.device)
                    else:
                        fL_cur = fL_all[env_i]
                        fR_cur = fR_all[env_i]
                        if press_require_any_contact:
                            press_contact_ok = torch.maximum(fL_cur, fR_cur) > press_contact_threshold
                        else:
                            press_contact_ok = torch.minimum(fL_cur, fR_cur) > press_contact_threshold

                        if stall_require_any_contact:
                            stall_contact_ok = torch.maximum(fL_cur, fR_cur) > stall_contact_threshold
                        else:
                            stall_contact_ok = torch.minimum(fL_cur, fR_cur) > stall_contact_threshold

                    # phase latch
                    if hasattr(base_env, "_grasp_success_given"):
                        press_phase = bool(base_env._grasp_success_given[env_i].item())
                    else:
                        press_phase = False

                    # width / close
                    if len(gripper_jids) == 1:
                        width_p = 2.0 * robot.data.joint_pos[env_i, gripper_jids[0]]
                    else:
                        width_p = robot.data.joint_pos[env_i, gripper_jids].sum()
                    closedness_p = 1.0 - torch.clamp(width_p / press_open_width, 0.0, 1.0)
                    press_close_ok = closedness_p > press_min_closedness

                    # handle pos / vel
                    handle_pos_cur = door.data.joint_pos[env_i, handle_jid]
                    handle_vel_raw_cur = door.data.joint_vel[env_i, handle_jid]

                    # dpos from cached prev handle pos (non-invasive: do not write cache here)
                    if hasattr(base_env, "_press_prev_handle_pos") and base_env._press_prev_handle_pos.shape[0] > env_i:
                        prev_handle_pos_cur = base_env._press_prev_handle_pos[env_i]
                        dpos_cur = (prev_handle_pos_cur - handle_pos_cur) if press_less_than else (handle_pos_cur - prev_handle_pos_cur)
                    else:
                        dpos_cur = torch.tensor(0.0, device=base_env.device)

                    prog_gate_cur = torch.tanh(torch.clamp(dpos_cur - float(press_pos_deadzone), min=0.0) / float(press_pos_scale))

                    if hasattr(base_env, "_press_vel_ema") and base_env._press_vel_ema.shape[0] > env_i:
                        vel_ema_cur = base_env._press_vel_ema[env_i]
                    else:
                        vel_ema_cur = handle_vel_raw_cur

                    desired_cur = (-vel_ema_cur) if press_less_than else vel_ema_cur
                    pos_cur = torch.clamp(desired_cur - float(press_vel_deadzone), min=0.0)
                    neg_cur = torch.clamp(-desired_cur - float(press_vel_deadzone), min=0.0)
                    r_pos_cur = torch.tanh(pos_cur / float(press_vel_scale)) * prog_gate_cur
                    r_neg_cur = torch.tanh(neg_cur / float(press_vel_scale))
                    r_cur = r_pos_cur - float(press_opposite_penalty) * r_neg_cur

                    press_gate = bool(press_phase) and bool(press_contact_ok.item()) and bool(press_close_ok.item())

                    # current stall (position-based) diagnosis
                    if stall_less_than:
                        stall_depth_cur = torch.clamp((handle_pos_cur - float(stall_pos)) / float(stall_pos_scale), min=0.0)
                    else:
                        stall_depth_cur = torch.clamp((float(stall_pos) - handle_pos_cur) / float(stall_pos_scale), min=0.0)

                    if hasattr(base_env, "_grasp_recent_ttl") and hasattr(base_env, "_grasp_recent_grace"):
                        stall_phase_cur = bool((base_env._grasp_recent_ttl[env_i] > 0).item())
                        stall_grace_ok_cur = bool((base_env._grasp_recent_grace[env_i] == 0).item())
                    else:
                        stall_phase_cur = False
                        stall_grace_ok_cur = False

                    stall_bad = stall_phase_cur and stall_grace_ok_cur and bool(stall_contact_ok.item()) and bool((stall_depth_cur > 0).item())

                    lbl_press_gate.text = f"press gate (env{HUD_ENV_INDEX}): {press_gate}"
                    lbl_press_phase.text = (
                        f"press phase / contact / close (env{HUD_ENV_INDEX}): "
                        f"{press_phase} / {bool(press_contact_ok.item())} / {bool(press_close_ok.item())}"
                    )
                    lbl_press_prog.text = (
                        f"press dpos / prog_gate (env{HUD_ENV_INDEX}): "
                        f"{float(dpos_cur.item()):.6f} / {float(prog_gate_cur.item()):.3f}"
                    )
                    lbl_press_vel.text = (
                        f"press vel_raw / vel_ema / desired (env{HUD_ENV_INDEX}): "
                        f"{float(handle_vel_raw_cur.item()):.6f} / {float(vel_ema_cur.item()):.6f} / {float(desired_cur.item()):.6f}"
                    )
                    lbl_press_shaping.text = (
                        f"press r_pos / r_neg / r (env{HUD_ENV_INDEX}): "
                        f"{float(r_pos_cur.item()):.4f} / {float(r_neg_cur.item()):.4f} / {float(r_cur.item()):.4f}"
                    )
                    lbl_stall.text = (
                        f"stall bad / depth (env{HUD_ENV_INDEX}): "
                        f"{stall_bad} / {float(stall_depth_cur.item()):.4f}"
                    )

                except Exception:
                    lbl_press_gate.text = f"press gate (env{HUD_ENV_INDEX}): N/A"
                    lbl_press_phase.text = f"press phase / contact / close (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_press_prog.text = f"press dpos / prog_gate (env{HUD_ENV_INDEX}): N/A / N/A"
                    lbl_press_vel.text = f"press vel_raw / vel_ema / desired (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_press_shaping.text = f"press r_pos / r_neg / r (env{HUD_ENV_INDEX}): N/A / N/A / N/A"
                    lbl_stall.text = f"stall bad / depth (env{HUD_ENV_INDEX}): N/A / N/A"

            _hud_stream = omni.kit.app.get_app().get_update_event_stream()
            _hud_sub = _hud_stream.create_subscription_to_pop(_on_hud_update)

        except Exception:
            _hud_win, _hud_sub = None, None

   # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    console_log_mode = os.environ.get("DOORBOT_CONSOLE_LOG_MODE", "stage_and_termination")
    if console_log_mode not in ("all", "stage_and_termination", "minimal"):
        raise ValueError(
            "DOORBOT_CONSOLE_LOG_MODE must be one of: all, stage_and_termination, minimal. "
            f"Got: {console_log_mode}"
        )
    runner = _patch_runner_console_log(runner, console_log_mode)

    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if should_load_checkpoint:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if resume_mode == "finetune":
            print("[INFO]: Resume mode: finetune warm-start (model/normalizers only, optimizer skipped, iteration=0).")
            _load_finetune_checkpoint(runner, resume_path, agent_cfg.device)
        else:
            print("[INFO]: Resume mode: full (model, optimizer, and iteration restored).")
            runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
