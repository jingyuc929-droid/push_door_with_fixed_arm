# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

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
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# --- Debug: force gripper open/close to validate action->target->drive path ---
parser.add_argument(
    "--force_gripper",
    action="store_true",
    default=False,
    help="Override the last action dimension (gripper) for a short duration to validate gripper targets/drives.",
)
parser.add_argument(
    "--force_gripper_seconds",
    type=float,
    default=1.0,
    help="Duration (seconds) for each forced gripper phase.",
)
parser.add_argument(
    "--force_gripper_pattern",
    type=str,
    default="close_open",
    choices=["close", "open", "close_open"],
    help="Pattern for forced gripper action.",
)
parser.add_argument(
    "--force_gripper_print_every",
    type=int,
    default=10,
    help="Print debug info every N env steps while playing.",
)

parser.add_argument(
    "--print_contact_forces",
    action="store_true",
    default=False,
    help="If set, print left/right contact sensor net force vectors and norms (env0) at the debug print frequency.",
)
parser.add_argument(
    "--left_contact_sensor",
    type=str,
    default="left_finger_contact",
    help="Scene name of the left finger contact sensor (ContactSensor).",
)
parser.add_argument(
    "--right_contact_sensor",
    type=str,
    default="right_finger_contact",
    help="Scene name of the right finger contact sensor (ContactSensor).",
)

# --- Lightweight rollout stats (reduce print spam) ---
parser.add_argument(
    "--stats_every",
    type=int,
    default=None,
    help="Print compact rollout stats every N env steps (0 disables). If not set, falls back to --force_gripper_print_every.",
)
parser.add_argument(
    "--stats_contact_threshold",
    type=float,
    default=0.5,
    help="Threshold (N) on handle-filtered contact force to count as 'any contact'.",
)
parser.add_argument(
    "--stats_min_sep",
    type=float,
    default=0.002,
    help="Minimum separation (in handle frame along grasp axis) to count wrap alignment.",
)
parser.add_argument(
    "--stats_open_width",
    type=float,
    default=0.088,
    help="Open width used to compute closedness. For X5 single gripper_joint, width is derived as 2*q.",
)
parser.add_argument(
    "--debug_fingers",
    action="store_true",
    default=False,
    help="Print detailed finger joint/force info for env0 at the stats frequency.",
)
parser.add_argument(
    "--debug_setup",
    action="store_true",
    default=False,
    help="Print actuator/joint mapping once at startup.",
)
parser.add_argument(
    "--record_success_trajectories",
    action="store_true",
    default=False,
    help="Record per-frame episode trajectories and save only successful episodes by default.",
)
parser.add_argument(
    "--success_door_open_threshold",
    type=float,
    default=0.30,
    help="Door-open threshold used to classify a recorded episode as successful.",
)
parser.add_argument(
    "--max_success_trajectories",
    type=int,
    default=5,
    help="Stop play after saving this many successful trajectories.",
)
parser.add_argument(
    "--trajectory_output_dir",
    type=str,
    default="logs/trajectories",
    help="Root directory for saved trajectory recordings.",
)
parser.add_argument(
    "--disable_staged_reset",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable staged reset before env creation. Use --no-disable_staged_reset to keep staged reset enabled.",
)
parser.add_argument(
    "--save_failed_trajectories",
    action="store_true",
    default=False,
    help="Also save failed episodes when recording trajectories.",
)
parser.add_argument(
    "--trajectory_format",
    choices=["npz", "csv", "both"],
    default="npz",
    help="Trajectory file format. NPZ is always recommended for waypoint loading.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.record_success_trajectories and "--agent" not in sys.argv:
    args_cli.agent = "rsl_rl_teacher_cfg_entry_point"
    print("[INFO] record_success_trajectories enabled: defaulting --agent to rsl_rl_teacher_cfg_entry_point")
# fallback: if stats_every is not provided, reuse the existing debug frequency flag
if args_cli.stats_every is None:
    args_cli.stats_every = args_cli.force_gripper_print_every
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import csv
import os
import time
from datetime import datetime
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import door_env.tasks  # noqa: F401
from door_env.tasks.manager_based.door_env.mdp.rewards import compute_stage1_grasp_quality


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.disable_staged_reset:
        if hasattr(env_cfg, "events") and hasattr(env_cfg.events, "staged_reset"):
            params = env_cfg.events.staged_reset.params
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0
            print("Staged reset disabled: True")
            print(f"p_grasp_start={params.get('p_grasp_start', 'N/A')}")
            print(f"p_unlock_start={params.get('p_unlock_start', 'N/A')}")
            if "p_opening_start" in params:
                print(f"p_opening_start={params.get('p_opening_start')}")
        else:
            print("[WARN] Staged reset disabled requested, but env_cfg.events.staged_reset was not found.")
            print("Staged reset disabled: False")
    else:
        print("Staged reset disabled: False")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        checkpoint_arg = os.path.expanduser(args_cli.checkpoint)
        if os.path.isabs(checkpoint_arg) or os.path.exists(checkpoint_arg):
            resume_path = retrieve_file_path(checkpoint_arg)
        else:
            candidate_path = os.path.join(log_root_path, agent_cfg.load_run, checkpoint_arg)
            if os.path.exists(candidate_path):
                resume_path = candidate_path
            else:
                resume_path = retrieve_file_path(checkpoint_arg)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
        
    # ---------------------------------------------------------------------
    # Debug/Stats helpers: query finger joint ids, body ids, sensors, and targets.
    # ---------------------------------------------------------------------
    robot = None
    door = None
    finger_joint_ids = None
    finger_body_ids = None  # (left_body_id, right_body_id)
    hand_body_id = None  # (left_body_id, right_body_id)
    handle_body_id = None
    q_target_field = None
    left_sensor = None
    right_sensor = None
    try:
        robot = base_env.scene["robot"]
        # door articulation (used for handle pose). If your asset name differs, update here.
        try:
            door = base_env.scene["door"]
        except Exception:
            door = None

        # contact sensors (we fetch them for stats even if not printing)
        try:
            left_sensor = base_env.scene[args_cli.left_contact_sensor]
        except Exception:
            left_sensor = None
        try:
            right_sensor = base_env.scene[args_cli.right_contact_sensor]
        except Exception:
            right_sensor = None
        if (left_sensor is None or right_sensor is None) and (args_cli.debug_setup or args_cli.print_contact_forces):
            print(
                f"[WARN] Contact sensor(s) not found. left='{args_cli.left_contact_sensor}' -> {left_sensor is not None}, "
                f"right='{args_cli.right_contact_sensor}' -> {right_sensor is not None}."
            )

        # joint name -> id mapping
        joint_names = None
        if hasattr(robot, "data") and hasattr(robot.data, "joint_names"):
            joint_names = list(robot.data.joint_names)
        elif hasattr(robot, "joint_names"):
            joint_names = list(robot.joint_names)

        if joint_names is not None:
            finger_joint_ids = []
            # X5 uses a single driven gripper joint. Prefer it explicitly.
            if "gripper_joint" in joint_names:
                finger_joint_ids = [joint_names.index("gripper_joint")]
            else:
                # Fallback for other robots: collect joints whose names suggest finger/gripper motion.
                for i, jn in enumerate(joint_names):
                    name = jn.lower()
                    if ("finger" in name) or ("gripper" in name):
                        finger_joint_ids.append(i)
            if len(finger_joint_ids) == 0:
                finger_joint_ids = None

        # body name -> id mapping (for wrap alignment stats)
        if hasattr(robot, "data") and hasattr(robot.data, "body_names"):
            bnames_r = list(robot.data.body_names)
            # Prefer true fingertip/pad bodies if present; otherwise fall back to finger links.
            left_candidates = ["left_pad", "left_finger", "link7"]
            right_candidates = ["right_pad", "right_finger", "link8"]
            left_bid = next((bnames_r.index(n) for n in left_candidates if n in bnames_r), None)
            right_bid = next((bnames_r.index(n) for n in right_candidates if n in bnames_r), None)
            if left_bid is not None and right_bid is not None:
                finger_body_ids = (left_bid, right_bid)
            if "link6" in bnames_r:
                hand_body_id = bnames_r.index("link6")
        if door is not None and hasattr(door, "data") and hasattr(door.data, "body_names"):
            bnames_d = list(door.data.body_names)
            if "handle_1" in bnames_d:
                handle_body_id = bnames_d.index("handle_1")

        # figure out which field stores joint position targets
        if hasattr(robot, "data"):
            if hasattr(robot.data, "joint_pos_target"):
                q_target_field = "joint_pos_target"
            elif hasattr(robot.data, "joint_pos_targets"):
                q_target_field = "joint_pos_targets"

        # print actuator coverage once (only if requested)
        if args_cli.debug_setup and hasattr(robot, "actuators"):
            print("[DEBUG] Actuator groups:", list(robot.actuators.keys()))
            for name, act in robot.actuators.items():
                jids = getattr(act, "joint_ids", None)
                jnames = getattr(act, "joint_names", None)
                jexpr = getattr(act, "joint_names_expr", None)
                print(f"  - {name}: joint_ids={jids} joint_names={jnames} expr={jexpr}")

            print(f"[DEBUG] Finger joint ids: {finger_joint_ids}")
            print(f"[DEBUG] Finger body ids: {finger_body_ids}, handle_body_id: {handle_body_id}")
            print(f"[DEBUG] Target field: {q_target_field}")

    except Exception as e:
        if args_cli.debug_setup:
            print(f"[DEBUG] Could not query robot joint/actuator info: {e}")
    # ---------------------------------------------------------------------
    # Small math helpers for stats (wxyz quaternions)
    # ---------------------------------------------------------------------
    def _quat_conjugate(q):
        return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)

    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack((w, x, y, z), dim=-1)

    def _quat_rotate(q, v):
        qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
        return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]

    def _filtered_force_norm(sensor):
        """Per-env norm of HANDLE-FILTERED force using force_matrix_w (safe-fail to zeros)."""
        fm = getattr(sensor.data, "force_matrix_w", None)
        if fm is None:
            return torch.zeros((env.num_envs,), device=env.unwrapped.device)
        if fm.ndim == 4:
            vec = fm.sum(dim=2)                        # [N,B,3]
            mag = torch.linalg.norm(vec, dim=-1)       # [N,B]
            return mag.max(dim=1).values               # [N]
        if fm.ndim == 3:
            vec = fm.sum(dim=1)                        # [N,3]
            return torch.linalg.norm(vec, dim=-1)      # [N]
        return torch.zeros((env.num_envs,), device=env.unwrapped.device)

    def _safe_unit(v, eps: float = 1e-8):
        return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


    def _compute_gripper_width(robot, joint_ids):
        """Return per-env derived gripper width.

        - single driven joint (X5): width = 2*q
        - multiple finger joints:   width = sum(q_i)
        """
        if joint_ids is None or len(joint_ids) == 0:
            return None
        if len(joint_ids) == 1:
            q = robot.data.joint_pos[:, joint_ids[0]]
            return 2.0 * q
        return robot.data.joint_pos[:, joint_ids].sum(dim=-1)

    def _compute_tip_gap(robot, finger_body_ids):
        """Approximate fingertip gap from the selected left/right finger bodies."""
        if finger_body_ids is None:
            return None
        lb, rb = finger_body_ids
        pL = robot.data.body_pos_w[:, lb, :]
        pR = robot.data.body_pos_w[:, rb, :]
        return torch.linalg.norm(pL - pR, dim=-1)


    def _best_signed_axis(v_local: torch.Tensor):
        """Return best matching signed basis axis for a 3D vector (single vector, shape [3])."""
        axes = [
            ("+x", torch.tensor([ 1.0,  0.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("-x", torch.tensor([-1.0,  0.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("+y", torch.tensor([ 0.0,  1.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("-y", torch.tensor([ 0.0, -1.0,  0.0], device=v_local.device, dtype=v_local.dtype)),
            ("+z", torch.tensor([ 0.0,  0.0,  1.0], device=v_local.device, dtype=v_local.dtype)),
            ("-z", torch.tensor([ 0.0,  0.0, -1.0], device=v_local.device, dtype=v_local.dtype)),
        ]
        best_name, best_dot = None, -1.0e9
        for name, axis in axes:
            dot = torch.dot(v_local, axis).item()
            if dot > best_dot:
                best_name, best_dot = name, dot
        return best_name, best_dot

    class SuccessTrajectoryRecorder:
        """Per-env episode recorder that persists successful trajectories."""

        def __init__(
            self,
            base_env,
            output_root: str,
            trajectory_format: str,
            success_threshold: float,
            max_success: int,
            save_failed: bool,
        ):
            import numpy as np

            self.np = np
            self.base_env = base_env
            self.num_envs = int(base_env.num_envs)
            self.device = base_env.device
            self.trajectory_format = trajectory_format
            self.success_threshold = float(success_threshold)
            self.max_success = int(max_success)
            self.save_failed = bool(save_failed)
            self.success_count = 0
            self.total_saved = 0
            self.buffers = [[] for _ in range(self.num_envs)]
            self.episode_ids = [0 for _ in range(self.num_envs)]
            self.warned_missing = set()

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.abspath(os.path.join(output_root, f"{stamp}_success_trajs"))
            os.makedirs(self.output_dir, exist_ok=True)
            self.manifest_path = os.path.join(self.output_dir, "manifest.csv")
            self.manifest_fields = [
                "file",
                "env_id",
                "episode_id",
                "success",
                "episode_length",
                "max_door_open",
                "final_door_open",
                "max_grasp_quality",
                "mean_stage1_grasp_quality",
                "mean_stage2_grasp_quality",
                "mean_stage1_closed_no_contact",
                "mean_stage2_closed_no_contact",
                "W1_idx",
                "W2_idx",
                "W3_idx",
                "W4_idx",
                "W5_idx",
                "W6_idx",
            ]
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.manifest_fields).writeheader()

            self.robot = self._scene_asset("robot")
            self.door = self._scene_asset("door")
            self.left_sensor = self._scene_asset(args_cli.left_contact_sensor, warn=False)
            self.right_sensor = self._scene_asset(args_cli.right_contact_sensor, warn=False)
            self.robot_joint_ids = self._joint_ids(self.robot, [f"joint{i}" for i in range(1, 7)], "robot arm joints")
            self.gripper_joint_ids = self._joint_ids(self.robot, ["gripper_joint"], "gripper_joint", required=False)
            self.door_joint_id = self._joint_ids(self.door, ["door_joint"], "door_joint", required=False)
            self.handle_joint_id = self._joint_ids(self.door, ["handle_joint"], "handle_joint", required=False)
            self.link6_body_id = self._body_id(self.robot, "link6", "link6")
            self.handle_body_id = self._body_id(self.door, "handle_1", "handle_1")
            self.left_body_id = self._body_id(self.robot, "link7", "left finger link7", required=False)
            self.right_body_id = self._body_id(self.robot, "link8", "right finger link8", required=False)
            self.arm_action_term = self._action_term("arm_action")
            self.gripper_action_term = self._action_term("gripper_action")

            self.quality_cfgs = None
            self._init_quality_cfgs()

            print(f"[TRAJ] Recording success trajectories to: {self.output_dir}")
            print(f"[TRAJ] success_door_open_threshold={self.success_threshold:.3f}, max_success={self.max_success}")

        def _warn_once(self, key: str, message: str):
            if key not in self.warned_missing:
                self.warned_missing.add(key)
                print(f"[TRAJ][WARN] {message}")

        def _scene_asset(self, name: str, warn: bool = True):
            try:
                return self.base_env.scene[name]
            except Exception:
                if warn:
                    self._warn_once(f"scene:{name}", f"Scene asset '{name}' not found; related fields use placeholders.")
                return None

        def _joint_ids(self, asset, names: list[str], label: str, required: bool = True):
            if asset is None or not hasattr(asset, "data") or not hasattr(asset.data, "joint_names"):
                if required:
                    self._warn_once(f"joint:{label}", f"Cannot resolve {label}; joint fields use NaN.")
                return []
            available = list(asset.data.joint_names)
            ids = [available.index(name) for name in names if name in available]
            if required and len(ids) != len(names):
                missing = [name for name in names if name not in available]
                self._warn_once(f"joint:{label}", f"Missing {label}: {missing}; available={available}")
            return ids

        def _body_id(self, asset, name: str, label: str, required: bool = True):
            if asset is None or not hasattr(asset, "data") or not hasattr(asset.data, "body_names"):
                if required:
                    self._warn_once(f"body:{label}", f"Cannot resolve {label}; pose fields use NaN.")
                return None
            available = list(asset.data.body_names)
            if name in available:
                return available.index(name)
            if required:
                self._warn_once(f"body:{label}", f"Missing body {name}; available={available}")
            return None

        def _action_term(self, name: str):
            action_manager = getattr(self.base_env, "action_manager", None)
            if action_manager is None:
                self._warn_once("action_manager", "action_manager not found; low-level action fields use NaN.")
                return None
            if hasattr(action_manager, "_terms") and name in action_manager._terms:
                return action_manager._terms[name]
            if hasattr(action_manager, "get_term"):
                try:
                    return action_manager.get_term(name)
                except Exception:
                    pass
            self._warn_once(f"action:{name}", f"Action term '{name}' not found; related fields use NaN.")
            return None

        def _init_quality_cfgs(self):
            try:
                cfgs = {
                    "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                    "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                    "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
                    "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
                    "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
                }
                for cfg in cfgs.values():
                    cfg.resolve(self.base_env.scene)
                self.quality_cfgs = cfgs
            except Exception as e:
                self.quality_cfgs = None
                self._warn_once("quality_cfgs", f"Could not initialize grasp-quality helper: {e}")

        def _nan_vec(self, dim: int):
            return [float("nan")] * dim

        def _tensor_vec(self, tensor, env_id: int, dim: int | None = None):
            if tensor is None:
                return self._nan_vec(dim or 1)
            try:
                value = tensor[env_id]
                if value.ndim == 0:
                    return [float(value.item())]
                return [float(x) for x in value.detach().cpu().flatten().tolist()]
            except Exception:
                return self._nan_vec(dim or 1)

        def _tensor_scalar(self, tensor, env_id: int, default=float("nan")):
            if tensor is None:
                return default
            try:
                return float(tensor[env_id].detach().cpu().item())
            except Exception:
                return default

        def _bool_scalar(self, tensor, env_id: int, default=False):
            if tensor is None:
                return bool(default)
            try:
                return bool(tensor[env_id].detach().cpu().item())
            except Exception:
                return bool(default)

        def _action_tensor(self, term, attr: str, dim: int):
            if term is None or not hasattr(term, attr):
                return torch.full((self.num_envs, dim), float("nan"), device=self.device)
            try:
                value = getattr(term, attr)
                if value.ndim == 1:
                    value = value.unsqueeze(-1)
                return value[:, :dim]
            except Exception as e:
                self._warn_once(f"action_attr:{attr}", f"Could not read action term attribute '{attr}': {e}")
                return torch.full((self.num_envs, dim), float("nan"), device=self.device)

        def _quality_terms(self):
            nan = torch.full((self.num_envs,), float("nan"), device=self.device)
            false = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            defaults = {
                "quality": nan,
                "contact_ok": false,
                "closedness": nan,
                "tcp_dist": nan,
                "f_left": nan,
                "f_right": nan,
                "f_min": nan,
                "f_max": nan,
                "closed_no_contact": false,
                "single_finger": false,
            }
            if self.quality_cfgs is None:
                return defaults
            try:
                terms = compute_stage1_grasp_quality(
                    self.base_env,
                    left_sensor_name=args_cli.left_contact_sensor,
                    right_sensor_name=args_cli.right_contact_sensor,
                    **self.quality_cfgs,
                )
                defaults.update(terms)
                return defaults
            except Exception as e:
                self._warn_once("quality_compute", f"Could not compute per-env grasp quality: {e}")
                return defaults

        def _batch_state(self, actions, rewards, dones, step_index: int):
            nan1 = torch.full((self.num_envs,), float("nan"), device=self.device)
            nan3 = torch.full((self.num_envs, 3), float("nan"), device=self.device)
            nan4 = torch.full((self.num_envs, 4), float("nan"), device=self.device)
            nan6 = torch.full((self.num_envs, 6), float("nan"), device=self.device)

            robot_q = self.robot.data.joint_pos[:, self.robot_joint_ids] if self.robot is not None and len(self.robot_joint_ids) == 6 else nan6
            robot_dq = self.robot.data.joint_vel[:, self.robot_joint_ids] if self.robot is not None and len(self.robot_joint_ids) == 6 else nan6
            gripper_q = (
                self.robot.data.joint_pos[:, self.gripper_joint_ids]
                if self.robot is not None and len(self.gripper_joint_ids) > 0
                else torch.full((self.num_envs, 1), float("nan"), device=self.device)
            )
            gripper_opening = gripper_q[:, 0] if gripper_q.ndim == 2 and gripper_q.shape[1] > 0 else nan1

            link6_pos = self.robot.data.body_pos_w[:, self.link6_body_id, :] if self.robot is not None and self.link6_body_id is not None else nan3
            link6_quat = self.robot.data.body_quat_w[:, self.link6_body_id, :] if self.robot is not None and self.link6_body_id is not None else nan4
            ee_offset = torch.tensor((0.1523, 0.0, 0.0), dtype=link6_pos.dtype, device=self.device).view(1, 3).repeat(self.num_envs, 1)
            ee_tcp_pos = link6_pos + _quat_rotate(link6_quat, ee_offset) if torch.isfinite(link6_pos).any() else nan3
            ee_tcp_quat = link6_quat

            handle_pos = self.door.data.body_pos_w[:, self.handle_body_id, :] if self.door is not None and self.handle_body_id is not None else nan3
            handle_quat = self.door.data.body_quat_w[:, self.handle_body_id, :] if self.door is not None and self.handle_body_id is not None else nan4
            handle_offset = torch.tensor((-0.08, 0.04, 0.01), dtype=handle_pos.dtype, device=self.device).view(1, 3).repeat(self.num_envs, 1)
            handle_target = handle_pos + _quat_rotate(handle_quat, handle_offset) if torch.isfinite(handle_pos).any() else nan3

            door_joint_pos = self.door.data.joint_pos[:, self.door_joint_id[0]] if self.door is not None and len(self.door_joint_id) == 1 else nan1
            door_joint_vel = self.door.data.joint_vel[:, self.door_joint_id[0]] if self.door is not None and len(self.door_joint_id) == 1 else nan1
            handle_joint_pos = self.door.data.joint_pos[:, self.handle_joint_id[0]] if self.door is not None and len(self.handle_joint_id) == 1 else nan1
            handle_joint_vel = self.door.data.joint_vel[:, self.handle_joint_id[0]] if self.door is not None and len(self.handle_joint_id) == 1 else nan1
            door_open = torch.clamp(door_joint_pos - 0.0, min=0.0)

            if hasattr(self.base_env, "_door_lock_mode"):
                door_lock_mode = self.base_env._door_lock_mode
                physical_unlocked = door_lock_mode == 2
            elif hasattr(self.base_env, "_door_unlocked"):
                door_lock_mode = torch.full((self.num_envs,), -1, device=self.device)
                physical_unlocked = self.base_env._door_unlocked.to(dtype=torch.bool)
            else:
                door_lock_mode = torch.full((self.num_envs,), -1, device=self.device)
                physical_unlocked = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

            grasp_success = (
                self.base_env._grasp_success_given.to(dtype=torch.bool)
                if hasattr(self.base_env, "_grasp_success_given")
                else torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
            )
            stage_id = torch.zeros((self.num_envs,), dtype=torch.int64, device=self.device)
            stage_id = torch.where(grasp_success & (~physical_unlocked), torch.ones_like(stage_id), stage_id)
            stage_id = torch.where(physical_unlocked, torch.full_like(stage_id, 2), stage_id)

            q_des = self._action_tensor(self.arm_action_term, "q_des", 6)
            applied_delta = self._action_tensor(self.arm_action_term, "applied_delta", 6)
            arm_raw = self._action_tensor(self.arm_action_term, "raw_actions", 6)
            gripper_raw = self._action_tensor(self.gripper_action_term, "raw_actions", 1)
            q_des_error = q_des - robot_q if torch.isfinite(q_des).any() and torch.isfinite(robot_q).any() else nan6

            quality = self._quality_terms()
            finger_mid_dist = nan1
            if self.robot is not None and self.door is not None and self.left_body_id is not None and self.right_body_id is not None and self.handle_body_id is not None:
                p_left = self.robot.data.body_pos_w[:, self.left_body_id, :]
                p_right = self.robot.data.body_pos_w[:, self.right_body_id, :]
                finger_mid = 0.5 * (p_left + p_right)
                finger_mid_dist = torch.linalg.norm(finger_mid - handle_target, dim=-1)

            near_ok = quality["tcp_dist"] <= 0.14
            close_ok = quality["closedness"] > 0.35
            left_force = quality["f_left"]
            right_force = quality["f_right"]

            return {
                "step_index": step_index,
                "reward": rewards,
                "done": dones,
                "robot_joint_pos": robot_q,
                "robot_joint_vel": robot_dq,
                "gripper_joint_pos": gripper_q,
                "gripper_opening": gripper_opening,
                "action_raw_or_policy": actions,
                "arm_action": actions[:, :6] if actions.shape[-1] >= 6 else arm_raw,
                "gripper_action": actions[:, 6:7] if actions.shape[-1] >= 7 else gripper_raw,
                "q_des": q_des,
                "q_des_error": q_des_error,
                "applied_delta": applied_delta,
                "ee_tcp_pos_w": ee_tcp_pos,
                "ee_tcp_quat_w": ee_tcp_quat,
                "link6_pos_w": link6_pos,
                "link6_quat_w": link6_quat,
                "handle_pos_w": handle_pos,
                "handle_quat_w": handle_quat,
                "handle_grasp_target_w": handle_target,
                "handle_joint_pos": handle_joint_pos,
                "handle_joint_vel": handle_joint_vel,
                "door_joint_pos": door_joint_pos,
                "door_joint_vel": door_joint_vel,
                "door_open": door_open,
                "grasp_success_given": grasp_success,
                "physical_unlocked": physical_unlocked,
                "door_lock_mode": door_lock_mode,
                "stage_id": stage_id,
                "grasp_quality": quality["quality"],
                "grasp_quality_gate": quality["contact_ok"] & close_ok & near_ok,
                "tcp_to_grasp_dist": quality["tcp_dist"],
                "finger_mid_to_grasp_dist": finger_mid_dist,
                "contact_ok": quality["contact_ok"],
                "close_ok": close_ok,
                "near_ok": near_ok,
                "closed_no_contact": quality["closed_no_contact"],
                "single_finger": quality["single_finger"],
                "left_contact_force": left_force,
                "right_contact_force": right_force,
                "f_min": quality["f_min"],
                "f_max": quality["f_max"],
            }

        def _frame_from_batch(self, batch: dict, env_id: int, episode_step: int):
            frame = {
                "step_index": int(batch["step_index"]),
                "episode_step": int(episode_step),
                "env_id": int(env_id),
                "sim_time": float(batch["step_index"] * self.base_env.step_dt),
            }
            scalar_keys = {
                "reward",
                "done",
                "gripper_opening",
                "handle_joint_pos",
                "handle_joint_vel",
                "door_joint_pos",
                "door_joint_vel",
                "door_open",
                "grasp_success_given",
                "physical_unlocked",
                "door_lock_mode",
                "stage_id",
                "grasp_quality",
                "grasp_quality_gate",
                "tcp_to_grasp_dist",
                "finger_mid_to_grasp_dist",
                "contact_ok",
                "close_ok",
                "near_ok",
                "closed_no_contact",
                "single_finger",
                "left_contact_force",
                "right_contact_force",
                "f_min",
                "f_max",
            }
            for key, value in batch.items():
                if key in ("step_index",):
                    continue
                if key in scalar_keys:
                    if torch.is_tensor(value) and value.dtype == torch.bool:
                        frame[key] = self._bool_scalar(value, env_id)
                    else:
                        frame[key] = self._tensor_scalar(value, env_id)
                else:
                    dim = 1
                    if key in ("robot_joint_pos", "robot_joint_vel", "arm_action", "q_des", "q_des_error", "applied_delta"):
                        dim = 6
                    elif key in ("ee_tcp_pos_w", "link6_pos_w", "handle_pos_w", "handle_grasp_target_w"):
                        dim = 3
                    elif key in ("ee_tcp_quat_w", "link6_quat_w", "handle_quat_w"):
                        dim = 4
                    frame[key] = self._tensor_vec(value, env_id, dim=dim)
            return frame

        def step(self, actions, rewards, dones, step_index: int):
            rewards = rewards if torch.is_tensor(rewards) else torch.as_tensor(rewards, device=self.device)
            dones = dones.to(dtype=torch.bool) if torch.is_tensor(dones) else torch.as_tensor(dones, dtype=torch.bool, device=self.device)
            batch = self._batch_state(actions.detach(), rewards.detach(), dones.detach(), step_index)
            for env_id in range(self.num_envs):
                frame = self._frame_from_batch(batch, env_id, len(self.buffers[env_id]))
                self.buffers[env_id].append(frame)
                if bool(dones[env_id].item()):
                    self._finish_episode(env_id)
            return self.success_count >= self.max_success

        def _episode_arrays(self, episode: list[dict]):
            arrays = {}
            if not episode:
                return arrays
            for key in episode[0].keys():
                values = [frame[key] for frame in episode]
                arrays[key] = self.np.asarray(values)
            return arrays

        def _first_true(self, values):
            idx = self.np.nonzero(self.np.asarray(values, dtype=bool))[0]
            return int(idx[0]) if idx.size > 0 else -1

        def _nearest_idx(self, mask, values, target):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmin(self.np.abs(values[valid] - float(target)))])

        def _argmin_idx(self, mask, values):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmin(values[valid])])

        def _argmax_idx(self, mask, values):
            mask = self.np.asarray(mask, dtype=bool)
            values = self.np.asarray(values, dtype=float)
            valid = self.np.nonzero(mask & self.np.isfinite(values))[0]
            if valid.size == 0:
                return -1
            return int(valid[self.np.argmax(values[valid])])

        def _waypoints(self, arrays: dict):
            stage = arrays.get("stage_id", self.np.zeros((0,), dtype=int))
            quality = arrays.get("grasp_quality", self.np.asarray([]))
            tcp_dist = arrays.get("tcp_to_grasp_dist", self.np.asarray([]))
            grasp = arrays.get("grasp_success_given", self.np.asarray([]))
            unlocked = arrays.get("physical_unlocked", self.np.asarray([]))
            handle = arrays.get("handle_joint_pos", self.np.asarray([]))
            door_open = arrays.get("door_open", self.np.asarray([]))

            w1 = self._argmin_idx(stage == 0, tcp_dist)
            first_grasp = self._first_true(grasp)
            if first_grasp >= 0:
                mask = self.np.zeros_like(grasp, dtype=bool)
                mask[first_grasp : min(len(mask), first_grasp + 11)] = True
                w2 = self._argmax_idx(mask, quality)
            else:
                w2 = -1
            w3 = self._nearest_idx(stage == 1, handle, -0.20)
            w4 = self._first_true(unlocked)
            w5 = self._nearest_idx(stage == 2, door_open, 0.15)
            success_cross = self.np.asarray(door_open, dtype=float) >= self.success_threshold
            w6 = self._first_true(success_cross)
            return {"W1": w1, "W2": w2, "W3": w3, "W4": w4, "W5": w5, "W6": w6}

        def _nanmean(self, values):
            values = self.np.asarray(values, dtype=float)
            if values.size == 0 or not self.np.isfinite(values).any():
                return float("nan")
            return float(self.np.nanmean(values))

        def _metrics(self, arrays: dict):
            door_open = self.np.asarray(arrays.get("door_open", []), dtype=float)
            quality = self.np.asarray(arrays.get("grasp_quality", []), dtype=float)
            stage = self.np.asarray(arrays.get("stage_id", []), dtype=int)
            closed_no_contact = self.np.asarray(arrays.get("closed_no_contact", []), dtype=float)
            max_door = float(self.np.nanmax(door_open)) if door_open.size else float("nan")
            final_door = float(door_open[-1]) if door_open.size else float("nan")
            return {
                "max_door_open": max_door,
                "final_door_open": final_door,
                "max_grasp_quality": float(self.np.nanmax(quality)) if quality.size and self.np.isfinite(quality).any() else float("nan"),
                "mean_stage1_grasp_quality": self._nanmean(quality[stage == 1]) if quality.size and stage.size else float("nan"),
                "mean_stage2_grasp_quality": self._nanmean(quality[stage == 2]) if quality.size and stage.size else float("nan"),
                "mean_stage1_closed_no_contact": self._nanmean(closed_no_contact[stage == 1]) if closed_no_contact.size and stage.size else float("nan"),
                "mean_stage2_closed_no_contact": self._nanmean(closed_no_contact[stage == 2]) if closed_no_contact.size and stage.size else float("nan"),
            }

        def _expanded_csv_row(self, frame: dict):
            row = {}
            for key, value in frame.items():
                arr = self.np.asarray(value)
                if arr.ndim == 0:
                    row[key] = value
                else:
                    flat = arr.flatten()
                    for idx, item in enumerate(flat):
                        row[f"{key}_{idx}"] = item
            return row

        def _save_csv(self, path: str, episode: list[dict]):
            rows = [self._expanded_csv_row(frame) for frame in episode]
            fieldnames = sorted({key for row in rows for key in row.keys()})
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        def _save_episode(self, env_id: int, episode_id: int, episode: list[dict], success: bool, metrics: dict, waypoints: dict):
            prefix = "success" if success else "failed"
            stem = f"{prefix}_env{env_id}_episode{episode_id}_door{metrics['max_door_open']:.3f}"
            arrays = self._episode_arrays(episode)
            arrays["waypoint_indices"] = self.np.asarray(
                [waypoints["W1"], waypoints["W2"], waypoints["W3"], waypoints["W4"], waypoints["W5"], waypoints["W6"]],
                dtype=self.np.int64,
            )
            arrays["waypoint_names"] = self.np.asarray(["W1_pregrasp", "W2_grasp", "W3_press", "W4_unlock", "W5_open_mid", "W6_success"])

            npz_path = os.path.join(self.output_dir, f"{stem}.npz")
            csv_path = os.path.join(self.output_dir, f"{stem}_frames.csv")
            main_file = npz_path
            if self.trajectory_format in ("npz", "both"):
                self.np.savez_compressed(npz_path, **arrays)
            if self.trajectory_format in ("csv", "both"):
                self._save_csv(csv_path, episode)
                if self.trajectory_format == "csv":
                    main_file = csv_path

            row = {
                "file": os.path.basename(main_file),
                "env_id": env_id,
                "episode_id": episode_id,
                "success": success,
                "episode_length": len(episode),
                **metrics,
                "W1_idx": waypoints["W1"],
                "W2_idx": waypoints["W2"],
                "W3_idx": waypoints["W3"],
                "W4_idx": waypoints["W4"],
                "W5_idx": waypoints["W5"],
                "W6_idx": waypoints["W6"],
            }
            with open(self.manifest_path, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.manifest_fields).writerow(row)

            self.total_saved += 1
            if success:
                self.success_count += 1
                print("Saved success trajectory:")
            else:
                print("Saved failed trajectory:")
            print(f"    {main_file}")
            print(f"    len={len(episode)}")
            print(f"    max_door_open={metrics['max_door_open']:.3f}")
            print(f"    mean_stage1_grasp_quality={metrics['mean_stage1_grasp_quality']:.4f}")
            print(f"    mean_stage2_grasp_quality={metrics['mean_stage2_grasp_quality']:.4f}")
            print(f"    waypoint_indices={waypoints}")

        def _finish_episode(self, env_id: int):
            episode = self.buffers[env_id]
            episode_id = self.episode_ids[env_id]
            self.episode_ids[env_id] += 1
            self.buffers[env_id] = []
            if not episode:
                return
            arrays = self._episode_arrays(episode)
            metrics = self._metrics(arrays)
            success = bool(metrics["max_door_open"] >= self.success_threshold)
            saved = False
            if success or self.save_failed:
                waypoints = self._waypoints(arrays)
                self._save_episode(env_id, episode_id, episode, success, metrics, waypoints)
                saved = True
            print(
                f"Episode finished env={env_id}, success={success}, len={len(episode)}, "
                f"max_door_open={metrics['max_door_open']:.3f}, saved={self.success_count}/{self.max_success}"
            )
            if saved and success:
                pass
    
        # ---------------------------------------------------------------------
    # GUI HUD: show handle_joint pos/vel/progress in a small window (no terminal prints)
    # ---------------------------------------------------------------------
    def _start_handle_joint_hud(
        base_env,
        joint_name: str = "handle_joint",
        env_index: int = 0,
        update_hz: float = 20.0,
        handle_start_pos: float = 0.0,
        handle_threshold: float = -0.2,
    ):
        """GUI HUD:
        - handle_joint pos/vel/progress
        - door_unlocked flag
        - door_joint pos
        - one-click calibration of align axes on current frame (env0)
        """
        try:
            import omni.ui as ui
            import omni.kit.app
        except Exception:
            return None, None

        door = base_env.scene["door"]
        robot = base_env.scene["robot"]

        # --- resolve handle_joint id ---
        jnames = list(door.data.joint_names)
        if joint_name in jnames:
            handle_jid = jnames.index(joint_name)
            resolved_handle = joint_name
        else:
            cand = [i for i, n in enumerate(jnames) if "handle" in n.lower()]
            handle_jid = cand[0] if cand else 0
            resolved_handle = jnames[handle_jid]

        # --- resolve door_joint id ---
        door_jid = jnames.index("door_joint") if "door_joint" in jnames else None

        # need body ids for calibration
        if finger_body_ids is None or hand_body_id is None or handle_body_id is None:
            # body ids unresolved -> still show basic HUD
            calib_available = False
        else:
            calib_available = True
            left_bid, right_bid = finger_body_ids

        # state for last calibration result
        calib_state = {
            "done": False,
            "gripper_open_axis_hand": "--",
            "gripper_open_axis_hand_dot": "--",
            "gripper_approach_axis_hand": "--",
            "gripper_approach_axis_hand_dot": "--",
            "grasp_axis": "--",
            "grasp_axis_dot": "--",
            "handle_approach_axis": "--",
            "handle_approach_axis_dot": "--",
            "spread_hand": "--",
            "spread_handle": "--",
            "app_hand": "--",
            "app_handle": "--",
        }

        def _axis_name_to_cfg_vec(name: str):
            mapping = {
                "+x": "(1.0, 0.0, 0.0)",
                "-x": "(-1.0, 0.0, 0.0)",
                "+y": "(0.0, 1.0, 0.0)",
                "-y": "(0.0, -1.0, 0.0)",
                "+z": "(0.0, 0.0, 1.0)",
                "-z": "(0.0, 0.0, -1.0)",
            }
            return mapping.get(name, "--")

        def _axis_name_to_index_and_sign(name: str):
            mapping = {
                "+x": ("0", "+"),
                "-x": ("0", "-"),
                "+y": ("1", "+"),
                "-y": ("1", "-"),
                "+z": ("2", "+"),
                "-z": ("2", "-"),
            }
            return mapping.get(name, ("--", "--"))

        def _calibrate_current_frame():
            if not calib_available:
                calib_state["done"] = True
                calib_state["gripper_open_axis_hand"] = "N/A"
                return

            try:
                # world poses for env_index only
                pL_w = robot.data.body_pos_w[env_index, left_bid, :]
                pR_w = robot.data.body_pos_w[env_index, right_bid, :]
                qG = robot.data.body_quat_w[env_index, hand_body_id, :]
                pH_w = door.data.body_pos_w[env_index, handle_body_id, :]
                qH = door.data.body_quat_w[env_index, handle_body_id, :]

                # 1) spread direction = true finger opening direction in world
                d_spread_w = _safe_unit((pL_w - pR_w).unsqueeze(0))[0]

                # expressed in hand frame
                d_spread_hand = _quat_rotate(_quat_conjugate(qG).unsqueeze(0), d_spread_w.unsqueeze(0))[0]
                open_name, open_dot = _best_signed_axis(d_spread_hand)

                # expressed in handle frame
                d_spread_handle = _quat_rotate(_quat_conjugate(qH).unsqueeze(0), d_spread_w.unsqueeze(0))[0]
                grasp_name, grasp_dot = _best_signed_axis(d_spread_handle)

                # 2) approach direction = from fingertip midpoint to handle grasp point
                handle_offset_h = torch.tensor([-0.09, 0.04, 0.01], device=pH_w.device, dtype=pH_w.dtype)
                p_grasp_w = pH_w + _quat_rotate(qH.unsqueeze(0), handle_offset_h.unsqueeze(0))[0]
                p_mid_w = 0.5 * (pL_w + pR_w)
                d_app_w = _safe_unit((p_grasp_w - p_mid_w).unsqueeze(0))[0]

                d_app_hand = _quat_rotate(_quat_conjugate(qG).unsqueeze(0), d_app_w.unsqueeze(0))[0]
                app_hand_name, app_hand_dot = _best_signed_axis(d_app_hand)

                d_app_handle = _quat_rotate(_quat_conjugate(qH).unsqueeze(0), d_app_w.unsqueeze(0))[0]
                app_handle_name, app_handle_dot = _best_signed_axis(d_app_handle)

                grasp_idx, grasp_sign = _axis_name_to_index_and_sign(grasp_name)
                handle_app_idx, handle_app_sign = _axis_name_to_index_and_sign(app_handle_name)

                calib_state["done"] = True
                calib_state["gripper_open_axis_hand"] = _axis_name_to_cfg_vec(open_name)
                calib_state["gripper_open_axis_hand_dot"] = f"{open_dot:.3f} ({open_name})"

                calib_state["gripper_approach_axis_hand"] = _axis_name_to_cfg_vec(app_hand_name)
                calib_state["gripper_approach_axis_hand_dot"] = f"{app_hand_dot:.3f} ({app_hand_name})"

                calib_state["grasp_axis"] = f"{grasp_idx}   sign={grasp_sign}"
                calib_state["grasp_axis_dot"] = f"{grasp_dot:.3f} ({grasp_name})"

                calib_state["handle_approach_axis"] = f"{handle_app_idx}   sign={handle_app_sign}"
                calib_state["handle_approach_axis_dot"] = f"{app_handle_dot:.3f} ({app_handle_name})"

                calib_state["spread_hand"] = f"[{d_spread_hand[0].item():+.3f}, {d_spread_hand[1].item():+.3f}, {d_spread_hand[2].item():+.3f}]"
                calib_state["spread_handle"] = f"[{d_spread_handle[0].item():+.3f}, {d_spread_handle[1].item():+.3f}, {d_spread_handle[2].item():+.3f}]"
                calib_state["app_hand"] = f"[{d_app_hand[0].item():+.3f}, {d_app_hand[1].item():+.3f}, {d_app_hand[2].item():+.3f}]"
                calib_state["app_handle"] = f"[{d_app_handle[0].item():+.3f}, {d_app_handle[1].item():+.3f}, {d_app_handle[2].item():+.3f}]"

            except Exception as e:
                calib_state["done"] = True
                calib_state["gripper_open_axis_hand"] = f"ERR: {e}"

        win = ui.Window("Door Play HUD", width=520, height=460)
        with win.frame:
            with ui.VStack(spacing=6):
                lbl_pos = ui.Label("pos: --")
                lbl_vel = ui.Label("vel: --")
                lbl_prog = ui.Label("progress: --")
                lbl_unlock = ui.Label("door_unlocked: --")
                lbl_door = ui.Label("door_joint: --")
                lbl_q = ui.Label("gripper_joint q(env0): --")
                lbl_width = ui.Label(f"derived width(env0), open_width={args_cli.stats_open_width:.3f}: --")
                lbl_closed = ui.Label(f"closedness(env0), open_width={args_cli.stats_open_width:.3f}: --")
                lbl_closed_mean = ui.Label(f"closed_mean(all envs), open_width={args_cli.stats_open_width:.3f}: --")
                lbl_tip_gap = ui.Label("tip_gap(env0): --")
                lbl_contact_l = ui.Label(f"left sensor |F| (env{env_index}): --")
                lbl_contact_r = ui.Label(f"right sensor |F| (env{env_index}): --")
                lbl_contact_ok = ui.Label(f"contact_ok both>{args_cli.stats_contact_threshold:.2f} (env{env_index}): --")
                ui.Separator()
                btn = ui.Button("Calibrate axes on current frame", clicked_fn=_calibrate_current_frame)
                lbl_hint = ui.Label("Pause on a good grasp frame, then click the button.")
                ui.Separator()
                lbl_open = ui.Label("gripper_open_axis_hand: --")
                lbl_open_dot = ui.Label("  score: --")
                lbl_app = ui.Label("gripper_approach_axis_hand: --")
                lbl_app_dot = ui.Label("  score: --")
                lbl_grasp = ui.Label("grasp_axis: --")
                lbl_grasp_dot = ui.Label("  score: --")
                lbl_happ = ui.Label("handle_approach_axis: --")
                lbl_happ_dot = ui.Label("  score: --")
                ui.Separator()
                lbl_spread_hand = ui.Label("spread@hand: --")
                lbl_spread_handle = ui.Label("spread@handle: --")
                lbl_app_hand = ui.Label("approach@hand: --")
                lbl_app_handle = ui.Label("approach@handle: --")

        dt = 1.0 / max(1e-3, float(update_hz))
        last_t = 0.0
        denom = float(handle_threshold - handle_start_pos) if abs(handle_threshold - handle_start_pos) > 1e-9 else None

        def _on_update(_evt):
            nonlocal last_t
            now = time.time()
            if now - last_t < dt:
                return
            last_t = now

            # handle joint state
            pos = float(door.data.joint_pos[env_index, handle_jid].item())
            vel = float(door.data.joint_vel[env_index, handle_jid].item())
            lbl_pos.text = f"pos: {pos:.6f}"
            lbl_vel.text = f"vel: {vel:.6f}"

            # progress
            if denom is None:
                prog = 0.0
            else:
                prog = (pos - float(handle_start_pos)) / denom
                prog = 0.0 if prog < 0.0 else (1.0 if prog > 1.0 else prog)
            lbl_prog.text = f"progress: {prog:.3f}   (0→{handle_threshold})"

            # unlocked
            unlocked_flag = None
            if hasattr(base_env, "_door_unlocked"):
                try:
                    unlocked_flag = bool(base_env._door_unlocked[env_index].item())
                except Exception:
                    unlocked_flag = None

            by_handle = (pos < float(handle_threshold)) if float(handle_threshold) < float(handle_start_pos) else (pos > float(handle_threshold))
            if unlocked_flag is None:
                lbl_unlock.text = f"door_unlocked: N/A   | by_handle: {by_handle}"
            else:
                lbl_unlock.text = f"door_unlocked: {unlocked_flag}   | by_handle: {by_handle}"

            # door joint
            if door_jid is None:
                lbl_door.text = "door_joint: N/A"
            else:
                dpos = float(door.data.joint_pos[env_index, door_jid].item())
                dvel = float(door.data.joint_vel[env_index, door_jid].item())
                lbl_door.text = f"door_joint: pos={dpos:.4f} vel={dvel:.4f}"

            # gripper width / closedness / tip gap
            try:
                if finger_joint_ids is not None and len(finger_joint_ids) >= 1:
                    width_all = _compute_gripper_width(robot, finger_joint_ids)
                    if width_all is not None:
                        width_env0 = width_all[env_index]
                        closed_env0 = 1.0 - torch.clamp(width_env0 / float(args_cli.stats_open_width), 0.0, 1.0)
                        closed_all = 1.0 - torch.clamp(width_all / float(args_cli.stats_open_width), 0.0, 1.0)
                        lbl_width.text = (
                            f"derived width(env0), open_width={args_cli.stats_open_width:.3f}: "
                            f"{float(width_env0.item()):.5f}"
                        )
                        lbl_closed.text = (
                            f"closedness(env0), open_width={args_cli.stats_open_width:.3f}: "
                            f"{float(closed_env0.item()):.3f}"
                        )
                        lbl_closed_mean.text = (
                            f"closed_mean(all envs), open_width={args_cli.stats_open_width:.3f}: "
                            f"{float(closed_all.mean().item()):.3f}"
                        )

                        if len(finger_joint_ids) == 1:
                            q_env0 = robot.data.joint_pos[env_index, finger_joint_ids[0]]
                            lbl_q.text = f"gripper_joint q(env0): {float(q_env0.item()):.5f}   (width=2*q)"
                        else:
                            q_env0 = robot.data.joint_pos[env_index, finger_joint_ids]
                            lbl_q.text = f"gripper joints q(env0): {[round(float(v), 5) for v in q_env0.tolist()]}"
                    else:
                        lbl_q.text = "gripper_joint q(env0): N/A"
                        lbl_width.text = f"derived width(env0), open_width={args_cli.stats_open_width:.3f}: N/A"
                        lbl_closed.text = f"closedness(env0), open_width={args_cli.stats_open_width:.3f}: N/A"
                        lbl_closed_mean.text = f"closed_mean(all envs), open_width={args_cli.stats_open_width:.3f}: N/A"
                else:
                    lbl_q.text = "gripper_joint q(env0): N/A"
                    lbl_width.text = f"derived width(env0), open_width={args_cli.stats_open_width:.3f}: N/A"
                    lbl_closed.text = f"closedness(env0), open_width={args_cli.stats_open_width:.3f}: N/A"
                    lbl_closed_mean.text = f"closed_mean(all envs), open_width={args_cli.stats_open_width:.3f}: N/A"

                tip_gap_all = _compute_tip_gap(robot, finger_body_ids)
                if tip_gap_all is not None:
                    lbl_tip_gap.text = f"tip_gap(env0): {float(tip_gap_all[env_index].item()):.5f}"
                else:
                    lbl_tip_gap.text = "tip_gap(env0): N/A"
            except Exception:
                lbl_q.text = "gripper_joint q(env0): ERR"
                lbl_width.text = f"derived width(env0), open_width={args_cli.stats_open_width:.3f}: ERR"
                lbl_closed.text = f"closedness(env0), open_width={args_cli.stats_open_width:.3f}: ERR"
                lbl_closed_mean.text = f"closed_mean(all envs), open_width={args_cli.stats_open_width:.3f}: ERR"
                lbl_tip_gap.text = "tip_gap(env0): ERR"

            # finger contact sensor readings / contact gate
            try:
                if left_sensor is None or right_sensor is None:
                    lbl_contact_l.text = f"left sensor |F| (env{env_index}): N/A"
                    lbl_contact_r.text = f"right sensor |F| (env{env_index}): N/A"
                    lbl_contact_ok.text = f"contact_ok both>{args_cli.stats_contact_threshold:.2f} (env{env_index}): N/A"
                else:
                    f_l = _filtered_force_norm(left_sensor)[env_index]
                    f_r = _filtered_force_norm(right_sensor)[env_index]
                    contact_ok = torch.minimum(f_l, f_r) > float(args_cli.stats_contact_threshold)
                    lbl_contact_l.text = f"left sensor |F| (env{env_index}): {float(f_l.item()):.4f}"
                    lbl_contact_r.text = f"right sensor |F| (env{env_index}): {float(f_r.item()):.4f}"
                    lbl_contact_ok.text = (
                        f"contact_ok both>{args_cli.stats_contact_threshold:.2f} "
                        f"(env{env_index}): {bool(contact_ok.item())}"
                    )
            except Exception:
                lbl_contact_l.text = f"left sensor |F| (env{env_index}): ERR"
                lbl_contact_r.text = f"right sensor |F| (env{env_index}): ERR"
                lbl_contact_ok.text = f"contact_ok both>{args_cli.stats_contact_threshold:.2f} (env{env_index}): ERR"

            # calibration text
            lbl_open.text = f"gripper_open_axis_hand: {calib_state['gripper_open_axis_hand']}"
            lbl_open_dot.text = f"  score: {calib_state['gripper_open_axis_hand_dot']}"
            lbl_app.text = f"gripper_approach_axis_hand: {calib_state['gripper_approach_axis_hand']}"
            lbl_app_dot.text = f"  score: {calib_state['gripper_approach_axis_hand_dot']}"
            lbl_grasp.text = f"grasp_axis: {calib_state['grasp_axis']}"
            lbl_grasp_dot.text = f"  score: {calib_state['grasp_axis_dot']}"
            lbl_happ.text = f"handle_approach_axis: {calib_state['handle_approach_axis']}"
            lbl_happ_dot.text = f"  score: {calib_state['handle_approach_axis_dot']}"
            lbl_spread_hand.text = f"spread@hand: {calib_state['spread_hand']}"
            lbl_spread_handle.text = f"spread@handle: {calib_state['spread_handle']}"
            lbl_app_hand.text = f"approach@hand: {calib_state['app_hand']}"
            lbl_app_handle.text = f"approach@handle: {calib_state['app_handle']}"

        stream = omni.kit.app.get_app().get_update_event_stream()
        sub = stream.create_subscription_to_pop(_on_update)
        return win, sub


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print("Deterministic policy: True")

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # forced gripper debug schedule
    force_steps = max(1, int(args_cli.force_gripper_seconds / max(1e-9, dt))) if args_cli.force_gripper else 0

    # reset environment
    obs = env.get_observations()
    trajectory_recorder = None
    if args_cli.record_success_trajectories:
        trajectory_recorder = SuccessTrajectoryRecorder(
            base_env=env.unwrapped,
            output_root=args_cli.trajectory_output_dir,
            trajectory_format=args_cli.trajectory_format,
            success_threshold=args_cli.success_door_open_threshold,
            max_success=args_cli.max_success_trajectories,
            save_failed=args_cli.save_failed_trajectories,
        )
    # ---------------------------------------------------------------------
    # Start HUD (GUI only)
    # ---------------------------------------------------------------------
    _hud_win, _hud_sub = None, None
    try:
        if not args_cli.headless:
            _hud_win, _hud_sub = _start_handle_joint_hud(
                base_env=env.unwrapped,           # base env
                joint_name="handle_joint",
                env_index=0,
                update_hz=20.0,                   # 10~30 都可以
                handle_start_pos=0.0,
                handle_threshold=-0.3,
            )
    except Exception:
        pass
    timestep = 0
    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # Optionally override the last action dim (gripper) for a short window.
            # Convention: +1 -> open, -1 -> close (Binary action will threshold internally).
            stage = "policy"
            a_policy0 = float(actions[0, -1].item()) if actions.numel() > 0 else 0.0
            if args_cli.force_gripper and actions.shape[-1] >= 1:
                close_val, open_val = -1.0, 1.0
                if args_cli.force_gripper_pattern == "close":
                    if step_count < force_steps:
                        actions[:, -1] = close_val
                        stage = "forced_close"
                elif args_cli.force_gripper_pattern == "open":
                    if step_count < force_steps:
                        actions[:, -1] = open_val
                        stage = "forced_open"
                else:  # close_open
                    if step_count < force_steps:
                        actions[:, -1] = close_val
                        stage = "forced_close"
                    elif step_count < 2 * force_steps:
                        actions[:, -1] = open_val
                        stage = "forced_open"

            # env stepping
            obs, rewards, dones, _ = env.step(actions)

            if trajectory_recorder is not None:
                if trajectory_recorder.step(actions, rewards, dones, step_count):
                    print(
                        f"[TRAJ] Reached max_success_trajectories={args_cli.max_success_trajectories}; stopping play."
                    )
                    break

            # -----------------------------------------------------------------
            # Compact rollout stats (defaults to every 50 steps, see --stats_every)
            # -----------------------------------------------------------------
            if args_cli.stats_every > 0 and (step_count % args_cli.stats_every == 0):
                try:
                    # gripper close ratio (after any forced override)
                    close_ratio = 0.0
                    if actions.shape[-1] >= 1:
                        close_ratio = float((actions[:, -1] < 0.0).float().mean().item())

                    # handle-only contact ratio (filtered)
                    any_contact_ratio = 0.0
                    f_any_mean = 0.0
                    if (left_sensor is not None) and (right_sensor is not None):
                        fL = _filtered_force_norm(left_sensor)
                        fR = _filtered_force_norm(right_sensor)
                        f_any = torch.maximum(fL, fR)
                        any_contact_ratio = float((f_any > args_cli.stats_contact_threshold).float().mean().item())
                        f_any_mean = float(f_any.mean().item())

                    # wrap alignment ratio + fingertip-mid distance mean
                    align_ratio = float("nan")
                    dist_mean = float("nan")
                    closed_mean = float("nan")
                    tip_gap_mean = float("nan")

                    if (robot is not None) and (door is not None) and (finger_body_ids is not None) and (handle_body_id is not None):
                        lb, rb = finger_body_ids
                        pL = robot.data.body_pos_w[:, lb, :]
                        pR = robot.data.body_pos_w[:, rb, :]
                        pH = door.data.body_pos_w[:, handle_body_id, :]
                        qH = door.data.body_quat_w[:, handle_body_id, :]

# fingertip-mid distance to grasp point (handle-frame offset)
                        pTip = 0.5 * (pL + pR)
                        off = torch.tensor([-0.09, 0.04, 0.01], device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)
                        pG = pH + _quat_rotate(qH, off)
                        dist = torch.linalg.norm(pTip - pG, dim=-1)
                        dist_mean = float(dist.mean().item())

                        tip_gap = torch.linalg.norm(pL - pR, dim=-1)
                        tip_gap_mean = float(tip_gap.mean().item())

                        # wrap alignment in handle frame (grasp_axis=2 => z)
                        L_h = _quat_rotate(_quat_conjugate(qH), pL - pH)
                        R_h = _quat_rotate(_quat_conjugate(qH), pR - pH)
                        side = (L_h[:, 2] * R_h[:, 2]) < 0.0
                        sep = torch.abs(L_h[:, 2] - R_h[:, 2]) > float(args_cli.stats_min_sep)
                        align = (side & sep).float()
                        align_ratio = float(align.mean().item())

                    # closedness mean from finger joints
                    if (robot is not None) and (finger_joint_ids is not None) and (len(finger_joint_ids) >= 1):
                        width = _compute_gripper_width(robot, finger_joint_ids)
                        if width is not None:
                            closed = 1.0 - torch.clamp(width / float(args_cli.stats_open_width), 0.0, 1.0)
                            closed_mean = float(closed.mean().item())

                    print(
                        f"[STATS] step={step_count} stage={stage} "
                        f"close_ratio={close_ratio:.2f} align_ratio={align_ratio:.2f} "
                        f"any_contact_ratio={any_contact_ratio:.2f} f_any_mean={f_any_mean:.2f} "
                        f"dist_mean={dist_mean:.3f} tip_gap_mean={tip_gap_mean:.3f} "
                        f"closed_mean={closed_mean:.2f}"
                    )

                    # optional: detailed env0 finger debug
                    if args_cli.debug_fingers and (robot is not None) and (finger_joint_ids is not None) and (len(finger_joint_ids) >= 1):
                        q0 = robot.data.joint_pos[0, finger_joint_ids]
                        dq0 = robot.data.joint_vel[0, finger_joint_ids]
                        width0 = _compute_gripper_width(robot, finger_joint_ids)[0]
                        if q_target_field is not None:
                            q_t_all = getattr(robot.data, q_target_field)
                            q_t0 = q_t_all[0, finger_joint_ids]
                            print(f"[DBG0] finger q={q0.tolist()} dq={dq0.tolist()} q_target={q_t0.tolist()} width={float(width0.item()):.5f}")
                        else:
                            print(f"[DBG0] finger q={q0.tolist()} dq={dq0.tolist()} width={float(width0.item()):.5f} (no q_target field)")

                        if args_cli.print_contact_forces and (left_sensor is not None) and (right_sensor is not None):
                            fL0 = float(_filtered_force_norm(left_sensor)[0].item())
                            fR0 = float(_filtered_force_norm(right_sensor)[0].item())
                            print(f"[DBG0] filtered_contact |L|={fL0:.3f} |R|={fR0:.3f} (threshold={args_cli.stats_contact_threshold:.2f})")

                except Exception as e:
                    print(f"[STATS] failed: {e}")

            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        step_count += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
