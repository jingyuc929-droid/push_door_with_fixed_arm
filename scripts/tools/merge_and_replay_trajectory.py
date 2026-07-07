from __future__ import annotations

"""Merge a locally edited preview trajectory back into a full success trajectory and replay it."""

import argparse
import glob
import math
import os
import tempfile
import time

import numpy as np
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description=(
        "Merge preview_trajectory.npz into an original success_xxx.npz and optionally replay the full trajectory.\n\n"
        "Examples:\n"
        "  python scripts/tools/merge_and_replay_trajectory.py --task Template-Door-Env-v0 "
        "--source_trajectory logs/trajectories/success_xxx.npz "
        "--preview_trajectory logs/trajectories/preview_trajectory.npz "
        "--output_trajectory logs/trajectories/full_preview_trajectory.npz --blend_radius 20 --replay\n\n"
        "  python scripts/tools/merge_and_replay_trajectory.py --task Template-Door-Env-v0 "
        "--source_trajectory logs/trajectories/success_xxx.npz "
        "--preview_trajectory logs/trajectories/preview_trajectory.npz "
        "--output_trajectory logs/trajectories/full_preview_trajectory.npz --start_frame 320 "
        "--blend_radius 20 --replay\n\n"
        "  python scripts/tools/merge_and_replay_trajectory.py --task Template-Door-Env-v0 "
        "--output_trajectory logs/trajectories/full_preview_trajectory.npz --only_replay --loop"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--source_trajectory", type=str, default=None, help="Original success_xxx.npz.")
parser.add_argument("--preview_trajectory", type=str, default=None, help="Edited preview_trajectory.npz.")
parser.add_argument(
    "--manual_waypoints",
    type=str,
    default=None,
    help="Optional manual_waypoints_v1.npz. If set, interpolate waypoints directly into source_trajectory.",
)
parser.add_argument("--output_trajectory", type=str, required=True, help="Merged full_preview_trajectory.npz.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Task name.")
parser.add_argument("--start_frame", type=int, default=None, help="Fallback insertion start frame if preview has no mapping.")
parser.add_argument("--blend_radius", type=int, default=20, help="Raised-cosine boundary blend radius.")
parser.add_argument("--overwrite_door_state", action="store_true", default=False, help="Also overwrite door/handle fields.")
parser.add_argument("--replay", action="store_true", default=False, help="Replay merged output after saving.")
parser.add_argument("--only_replay", action="store_true", default=False, help="Only replay output_trajectory; skip merge.")
parser.add_argument("--playback_dt", type=float, default=0.03, help="Wall-clock delay per replay frame.")
parser.add_argument("--start_replay_frame", type=int, default=0, help="Replay start frame.")
parser.add_argument("--end_replay_frame", type=int, default=-1, help="Replay end frame inclusive; -1 means trajectory end.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop replay.")
parser.add_argument(
    "--draw_debug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw ee_tcp, handle target, finger midpoint, and TCP paths.",
)
parser.add_argument("--save_video", type=str, default=None, help="Optional replay video output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

simulation_app = None
carb = None
gym = None
omni = None
torch = None
SceneEntityCfg = None
DoorEnvEnvCfg = None
compute_stage1_grasp_quality = None


def _launch_isaac_if_needed():
    global simulation_app, carb, gym, omni, torch, SceneEntityCfg, DoorEnvEnvCfg, compute_stage1_grasp_quality
    if simulation_app is not None:
        return
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import carb as _carb
    import gymnasium as _gym
    import omni as _omni
    import omni.appwindow  # noqa: F401
    import torch as _torch
    from isaaclab.managers import SceneEntityCfg as _SceneEntityCfg

    import door_env.tasks  # noqa: F401
    from door_env.tasks.manager_based.door_env.door_env_env_cfg import DoorEnvEnvCfg as _DoorEnvEnvCfg
    from door_env.tasks.manager_based.door_env.mdp.rewards import compute_stage1_grasp_quality as _compute_stage1_grasp_quality

    carb = _carb
    gym = _gym
    omni = _omni
    torch = _torch
    SceneEntityCfg = _SceneEntityCfg
    DoorEnvEnvCfg = _DoorEnvEnvCfg
    compute_stage1_grasp_quality = _compute_stage1_grasp_quality


DEFAULT_EDIT_FIELDS = ("robot_joint_pos", "gripper_joint_pos")
DERIVED_FIELDS = ("ee_tcp_pos_w", "tcp_to_grasp_dist", "finger_mid_to_grasp_dist", "grasp_quality")
DOOR_FIELDS = ("door_joint_pos", "handle_joint_pos", "door_open", "door_joint_vel", "handle_joint_vel")


def _key_name(key) -> str:
    raw = str(key).split(".")[-1].upper()
    aliases = {
        "SPACEBAR": "SPACE",
        "LEFT_ARROW": "LEFT",
        "ARROW_LEFT": "LEFT",
        "RIGHT_ARROW": "RIGHT",
        "ARROW_RIGHT": "RIGHT",
    }
    return aliases.get(raw, raw)


def _resolve_input_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    candidates = [expanded] if os.path.isabs(expanded) else [os.path.abspath(expanded), os.path.join(PROJECT_ROOT, expanded)]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate

    parent = os.path.dirname(path)
    search_dirs = [os.path.abspath(parent), os.path.join(PROJECT_ROOT, parent)] if parent else [os.getcwd(), PROJECT_ROOT]
    suggestions = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            suggestions.extend(glob.glob(os.path.join(search_dir, "*.npz")))
    message = [f"Input file not found: {path}", "Tried:"]
    message.extend(f"  - {os.path.abspath(candidate)}" for candidate in candidates)
    if suggestions:
        message.append("Nearby candidate files:")
        message.extend(f"  - {item}" for item in sorted(dict.fromkeys(suggestions))[:10])
    raise FileNotFoundError("\n".join(message))


def _resolve_output_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(PROJECT_ROOT, expanded))


def _load_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    return {key: np.asarray(data[key]) for key in data.files}


def _trajectory_len(data: dict, label: str) -> int:
    if "robot_joint_pos" not in data:
        raise KeyError(f"{label} missing required field 'robot_joint_pos'.")
    arr = np.asarray(data["robot_joint_pos"])
    if arr.ndim == 0:
        raise ValueError(f"{label} field 'robot_joint_pos' must be time-indexed.")
    return int(arr.shape[0])


def _preview_global_indices(preview: dict, preview_len: int, source_len: int, cli_start_frame: int | None) -> np.ndarray:
    if "global_frame_indices" in preview:
        indices = np.asarray(preview["global_frame_indices"], dtype=np.int64).reshape(-1)
        source = "global_frame_indices"
    elif "source_frame_indices" in preview:
        indices = np.asarray(preview["source_frame_indices"], dtype=np.int64).reshape(-1)
        source = "source_frame_indices"
    elif "start_frame" in preview:
        start = int(np.asarray(preview["start_frame"]).reshape(-1)[0])
        indices = np.arange(start, start + preview_len, dtype=np.int64)
        source = "preview start_frame"
    elif cli_start_frame is not None:
        indices = np.arange(int(cli_start_frame), int(cli_start_frame) + preview_len, dtype=np.int64)
        source = "--start_frame"
    else:
        raise ValueError("no global_frame_indices/source_frame_indices/start_frame; pass --start_frame.")

    if indices.shape[0] != preview_len:
        raise ValueError(
            f"Preview frame mapping length mismatch from {source}: got {indices.shape[0]}, expected {preview_len}."
        )
    if np.any(indices < 0):
        bad = indices[np.nonzero(indices < 0)[0][0]]
        raise ValueError(f"Preview frame mapping contains negative frame index: {int(bad)}.")
    if np.any(indices >= source_len):
        bad = indices[np.nonzero(indices >= source_len)[0][0]]
        raise ValueError(f"Preview frame mapping contains out-of-range frame index {int(bad)} for source_len={source_len}.")
    return indices


def _continuous_segments(indices: np.ndarray):
    if indices.size == 0:
        return
    start = 0
    for i in range(1, len(indices)):
        if int(indices[i]) != int(indices[i - 1]) + 1:
            yield start, i
            start = i
    yield start, len(indices)


def _blend_alpha(length: int, requested_radius: int) -> np.ndarray:
    alpha = np.ones(length, dtype=np.float32)
    radius = min(max(int(requested_radius), 0), length // 2)
    if radius <= 0:
        return alpha

    for pos in range(radius + 1):
        alpha[pos] = min(alpha[pos], 0.5 * (1.0 - math.cos(math.pi * pos / radius)))
    right_start = length - radius - 1
    for pos in range(right_start, length):
        k = pos - right_start
        alpha[pos] = min(alpha[pos], 0.5 * (1.0 + math.cos(math.pi * k / radius)))
    return alpha


def _merge_continuous_field(out: dict, source: dict, preview: dict, field: str, indices: np.ndarray, blend_radius: int):
    src_arr = np.asarray(source[field])
    prev_arr = np.asarray(preview[field])
    if prev_arr.shape[0] != indices.shape[0]:
        raise ValueError(f"Preview field '{field}' length {prev_arr.shape[0]} does not match mapping length {indices.shape[0]}.")
    if src_arr.ndim != prev_arr.ndim or src_arr.shape[1:] != prev_arr.shape[1:]:
        raise ValueError(f"Field '{field}' shape mismatch: source {src_arr.shape}, preview {prev_arr.shape}.")

    result = np.asarray(out[field]).copy()
    for p0, p1 in _continuous_segments(indices):
        seg_indices = indices[p0:p1]
        alpha = _blend_alpha(p1 - p0, blend_radius).reshape((-1,) + (1,) * (prev_arr.ndim - 1))
        result[seg_indices] = (1.0 - alpha) * src_arr[seg_indices] + alpha * prev_arr[p0:p1]
    out[field] = result.astype(src_arr.dtype, copy=False)


def _overwrite_indexed_field(out: dict, source: dict, preview: dict, field: str, indices: np.ndarray):
    src_arr = np.asarray(source[field])
    prev_arr = np.asarray(preview[field])
    if prev_arr.shape[0] != indices.shape[0]:
        raise ValueError(f"Preview field '{field}' length {prev_arr.shape[0]} does not match mapping length {indices.shape[0]}.")
    if src_arr.ndim != prev_arr.ndim or src_arr.shape[1:] != prev_arr.shape[1:]:
        raise ValueError(f"Field '{field}' shape mismatch: source {src_arr.shape}, preview {prev_arr.shape}.")
    result = np.asarray(out[field]).copy()
    result[indices] = prev_arr
    out[field] = result.astype(src_arr.dtype, copy=False)


def _joint_jump_stats(robot_joint_pos: np.ndarray) -> tuple[float, float]:
    robot_joint_pos = np.asarray(robot_joint_pos, dtype=np.float32)
    if robot_joint_pos.shape[0] < 2:
        return 0.0, 0.0
    jumps = np.abs(np.diff(robot_joint_pos, axis=0))
    per_frame = np.max(jumps, axis=1)
    return float(np.max(per_frame)), float(np.mean(per_frame))


def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _manual_prefixes(manual: dict) -> list[str]:
    if "waypoint_names" in manual:
        prefixes = [str(name).split("_", 1)[0] for name in np.asarray(manual["waypoint_names"]).reshape(-1)]
    else:
        prefixes = sorted(
            key.split("_", 1)[0]
            for key in manual
            if key.endswith("_robot_joint_pos") and "_" in key
        )
    return [prefix for prefix in prefixes if f"{prefix}_robot_joint_pos" in manual and f"{prefix}_gripper_joint_pos" in manual]


def _load_manual_preview(manual_path: str, source_len: int) -> tuple[dict, np.ndarray]:
    manual = _load_npz(manual_path)
    prefixes = _manual_prefixes(manual)
    if len(prefixes) < 2:
        raise ValueError("manual_waypoints must contain at least two waypoint robot/gripper entries.")
    if "source_frame_indices" in manual:
        waypoint_frames = np.asarray(manual["source_frame_indices"], dtype=np.int64).reshape(-1)
    else:
        frames = []
        for prefix in prefixes:
            key = f"{prefix}_source_frame_index"
            if key not in manual:
                raise ValueError("manual_waypoints missing source_frame_indices and per-waypoint *_source_frame_index fields.")
            frames.append(int(np.asarray(manual[key]).reshape(-1)[0]))
        waypoint_frames = np.asarray(frames, dtype=np.int64)
    if waypoint_frames.shape[0] != len(prefixes):
        raise ValueError(
            f"manual waypoint frame count mismatch: {waypoint_frames.shape[0]} source frames for {len(prefixes)} waypoints."
        )
    if np.any(np.diff(waypoint_frames) <= 0):
        raise ValueError(f"manual source_frame_indices must be strictly increasing, got {waypoint_frames.tolist()}.")
    if np.any(waypoint_frames < 0) or np.any(waypoint_frames >= source_len):
        raise ValueError(f"manual source_frame_indices out of source range 0..{source_len - 1}: {waypoint_frames.tolist()}.")

    robot_wps = np.stack([np.asarray(manual[f"{prefix}_robot_joint_pos"], dtype=np.float32).reshape(-1) for prefix in prefixes], axis=0)
    gripper_wps = np.stack([np.asarray(manual[f"{prefix}_gripper_joint_pos"], dtype=np.float32).reshape(-1) for prefix in prefixes], axis=0)
    if robot_wps.shape[1] != 6:
        raise ValueError(f"manual robot_joint_pos must have dim 6, got {robot_wps.shape}.")

    global_indices = np.arange(int(waypoint_frames[0]), int(waypoint_frames[-1]) + 1, dtype=np.int64)
    robot = np.empty((len(global_indices), robot_wps.shape[1]), dtype=np.float32)
    gripper = np.empty((len(global_indices), gripper_wps.shape[1]), dtype=np.float32)

    for out_i, frame in enumerate(global_indices):
        seg = int(np.searchsorted(waypoint_frames, frame, side="right") - 1)
        seg = int(np.clip(seg, 0, len(waypoint_frames) - 2))
        denom = max(1, int(waypoint_frames[seg + 1] - waypoint_frames[seg]))
        t = float(frame - waypoint_frames[seg]) / float(denom)
        if len(prefixes) >= 4:
            i0 = max(seg - 1, 0)
            i1 = seg
            i2 = seg + 1
            i3 = min(seg + 2, len(prefixes) - 1)
            robot[out_i] = _catmull_rom(robot_wps[i0], robot_wps[i1], robot_wps[i2], robot_wps[i3], t)
            gripper[out_i] = _catmull_rom(gripper_wps[i0], gripper_wps[i1], gripper_wps[i2], gripper_wps[i3], t)
        else:
            robot[out_i] = (1.0 - t) * robot_wps[seg] + t * robot_wps[seg + 1]
            gripper[out_i] = (1.0 - t) * gripper_wps[seg] + t * gripper_wps[seg + 1]

    preview = {
        "robot_joint_pos": robot,
        "gripper_joint_pos": gripper,
        "global_frame_indices": global_indices,
        "manual_waypoint_frames": waypoint_frames,
        "manual_waypoint_names": np.asarray(prefixes),
    }
    return preview, waypoint_frames


def _merge_trajectories(source_path: str, preview_path: str | None, output_path: str) -> tuple[dict, dict]:
    source = _load_npz(source_path)
    source_len = _trajectory_len(source, "source_trajectory")
    if args_cli.manual_waypoints is not None:
        manual_path = _resolve_input_path(args_cli.manual_waypoints)
        preview, manual_frames = _load_manual_preview(manual_path, source_len)
        preview_path_for_metadata = manual_path
        print(f"[INFO] manual_waypoints resolved to: {manual_path}")
        print(f"[MERGE] manual waypoint source frames: {manual_frames.tolist()}")
    else:
        preview = _load_npz(preview_path)
        preview_path_for_metadata = preview_path
    preview_len = _trajectory_len(preview, "preview_trajectory")
    indices = _preview_global_indices(preview, preview_len, source_len, args_cli.start_frame)

    out = {key: np.asarray(value).copy() for key, value in source.items()}
    edited_fields = []

    before_max, before_mean = _joint_jump_stats(source["robot_joint_pos"])
    for field in DEFAULT_EDIT_FIELDS:
        if field not in source:
            raise KeyError(f"source_trajectory missing required editable field '{field}'.")
        if field not in preview:
            raise KeyError(f"preview_trajectory missing required editable field '{field}'.")
        _merge_continuous_field(out, source, preview, field, indices, args_cli.blend_radius)
        edited_fields.append(field)

    derived_fields_stale = True
    for field in DERIVED_FIELDS:
        if field in source and field in preview:
            _overwrite_indexed_field(out, source, preview, field, indices)
            edited_fields.append(field)

    if args_cli.overwrite_door_state:
        for field in DOOR_FIELDS:
            if field in source and field in preview:
                _overwrite_indexed_field(out, source, preview, field, indices)
                edited_fields.append(field)

    after_max, after_mean = _joint_jump_stats(out["robot_joint_pos"])
    metadata = {
        "merged_from_source": np.asarray(source_path),
        "merged_from_preview": np.asarray(preview_path_for_metadata),
        "merge_input_mode": np.asarray("manual_waypoints" if args_cli.manual_waypoints is not None else "preview_trajectory"),
        "merge_global_frame_indices": indices.astype(np.int64),
        "merge_start_frame": np.asarray([int(indices.min())], dtype=np.int64),
        "merge_end_frame": np.asarray([int(indices.max())], dtype=np.int64),
        "blend_radius": np.asarray([int(args_cli.blend_radius)], dtype=np.int64),
        "overwrite_door_state": np.asarray([bool(args_cli.overwrite_door_state)]),
        "edited_fields": np.asarray(edited_fields),
        "derived_fields_stale": np.asarray([derived_fields_stale]),
        "max_abs_joint_frame_jump_before": np.asarray([before_max], dtype=np.float32),
        "mean_abs_joint_frame_jump_before": np.asarray([before_mean], dtype=np.float32),
        "max_abs_joint_frame_jump_after": np.asarray([after_max], dtype=np.float32),
        "mean_abs_joint_frame_jump_after": np.asarray([after_mean], dtype=np.float32),
    }
    out.update(metadata)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **out)

    print("[MERGE] source_len:", source_len)
    print("[MERGE] preview_len:", preview_len)
    print(f"[MERGE] global range: {int(indices.min())}..{int(indices.max())}")
    print("[MERGE] blend_radius:", int(args_cli.blend_radius))
    print("[MERGE] edited_fields:", ", ".join(edited_fields))
    print(f"[MERGE] max_abs_joint_frame_jump_before={before_max:.6f} mean={before_mean:.6f}")
    print(f"[MERGE] max_abs_joint_frame_jump_after={after_max:.6f} mean={after_mean:.6f}")
    if after_max > 0.10:
        print(f"[WARN] max_abs_joint_frame_jump_after_merge is {after_max:.4f} rad > 0.10 rad.")
    print("[MERGE] output:", output_path)
    return out, metadata


def _joint_ids(asset, names: list[str]):
    available = list(asset.data.joint_names)
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"Missing joints {missing}. Available joints: {available}")
    return [available.index(name) for name in names]


def _body_id(asset, name: str):
    available = list(asset.data.body_names)
    if name not in available:
        raise RuntimeError(f"Missing body {name!r}. Available bodies: {available}")
    return available.index(name)


def _get_action_term(env, name: str):
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    if hasattr(action_manager, "_terms") and name in action_manager._terms:
        return action_manager._terms[name]
    if hasattr(action_manager, "get_term"):
        try:
            return action_manager.get_term(name)
        except Exception:
            return None
    return None


def quat_conjugate(q):
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_rotate(q, v):
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


class ReplayController:
    def __init__(self, start_frame: int):
        self.paused = False
        self.step_delta = 0
        self.reset_requested = False
        self.quit_requested = False
        self.help_requested = False
        self.start_frame = int(start_frame)
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        self.input_iface = carb.input.acquire_input_interface()
        self.sub = self.input_iface.subscribe_to_keyboard_events(self.keyboard, self._on_keyboard_event)

    def close(self):
        if getattr(self, "sub", None) is not None:
            self.input_iface.unsubscribe_to_keyboard_events(self.keyboard, self.sub)
            self.sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        event_type = str(event.type).split(".")[-1].upper()
        if event_type not in {"KEY_PRESS", "KEY_REPEAT"}:
            return True
        key = _key_name(event.input)
        if key == "SPACE":
            self.paused = not self.paused
            print(f"[REPLAY] paused={self.paused}")
        elif key == "LEFT":
            self.step_delta = -1
            self.paused = True
        elif key == "RIGHT":
            self.step_delta = 1
            self.paused = True
        elif key == "R":
            self.reset_requested = True
        elif key == "Q":
            self.quit_requested = True
        elif key == "H":
            self.help_requested = True
        return True


class VideoRecorder:
    def __init__(self, output_path: str | None, playback_dt: float):
        self.output_path = output_path
        self.playback_dt = playback_dt
        self.frames = []
        self.enabled = output_path is not None
        self.warned = False
        self.tmpdir = tempfile.mkdtemp(prefix="merge_replay_") if self.enabled else None

    def capture(self, frame_index: int):
        if not self.enabled:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            path = os.path.join(self.tmpdir, f"frame_{frame_index:06d}.png")
            capture_viewport_to_file(get_active_viewport(), path)
            self.frames.append(path)
        except Exception as exc:
            if not self.warned:
                print(f"[WARN] save_video capture unavailable; continuing without video: {exc}")
                self.warned = True
            self.enabled = False

    def save(self):
        if not self.enabled or not self.frames:
            return
        try:
            import imageio.v2 as imageio

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            fps = max(1, int(round(1.0 / max(float(self.playback_dt), 1.0e-6))))
            with imageio.get_writer(self.output_path, fps=fps) as writer:
                for path in self.frames:
                    if os.path.exists(path):
                        writer.append_data(imageio.imread(path))
            print(f"[VIDEO] saved to: {self.output_path}")
        except Exception as exc:
            print(f"[WARN] Could not save video: {exc}")


class FullTrajectoryReplay:
    def __init__(self, env, traj: dict, source_tcp: np.ndarray | None, max_jump_after: float):
        self.env = env
        self.robot = env.scene["robot"]
        self.door = env.scene["door"]
        self.traj = traj
        self.source_tcp = source_tcp
        self.max_jump_after = max_jump_after
        self.arm_joint_ids = _joint_ids(self.robot, [f"joint{i}" for i in range(1, 7)])
        self.gripper_joint_ids = _joint_ids(self.robot, ["gripper_joint"])
        self.door_joint_id = _joint_ids(self.door, ["door_joint"])[0]
        self.handle_joint_id = _joint_ids(self.door, ["handle_joint"])[0]
        self.link6_id = _body_id(self.robot, "link6")
        self.handle_body_id = _body_id(self.door, "handle_1")
        self.left_body_id = _body_id(self.robot, "link7")
        self.right_body_id = _body_id(self.robot, "link8")
        self.arm_action = _get_action_term(env, "arm_action")
        self.quality_cfgs = self._make_quality_cfgs()
        self.draw = self._make_draw() if args_cli.draw_debug else None
        self.preview_tcp_points = []
        self.warned_quality = False

    @property
    def length(self):
        return _trajectory_len(self.traj, "output_trajectory")

    def _make_quality_cfgs(self):
        try:
            cfgs = {
                "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
                "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
                "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            }
            for cfg in cfgs.values():
                cfg.resolve(self.env.scene)
            return cfgs
        except Exception as exc:
            print(f"[WARN] grasp_quality helper unavailable: {exc}")
            return None

    def _make_draw(self):
        try:
            from isaacsim.util.debug_draw import _debug_draw

            return _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:
            print(f"[WARN] debug draw unavailable: {exc}")
            return None

    def _sync_action_targets(self, q_arm: torch.Tensor):
        if self.arm_action is None:
            return
        if hasattr(self.arm_action, "_q_des"):
            self.arm_action._q_des[0, : len(self.arm_joint_ids)] = q_arm
        if hasattr(self.arm_action, "_dq_des"):
            self.arm_action._dq_des[0, : len(self.arm_joint_ids)] = 0.0
        if hasattr(self.arm_action, "_tau_ff"):
            self.arm_action._tau_ff[0, : len(self.arm_joint_ids)] = 0.0
        if hasattr(self.arm_action, "_applied_delta"):
            self.arm_action._applied_delta[0, : len(self.arm_joint_ids)] = 0.0
        if hasattr(self.arm_action, "_pending_q_des_sync"):
            self.arm_action._pending_q_des_sync[0] = False

    def _frame_field(self, key: str, index: int, default=None):
        if key not in self.traj:
            return default
        arr = np.asarray(self.traj[key])
        if arr.ndim == 0 or index >= arr.shape[0]:
            return default
        return arr[index]

    def apply_frame(self, index: int):
        index = int(np.clip(index, 0, self.length - 1))
        device = self.env.device
        robot_q = self.robot.data.joint_pos[0].clone()
        robot_dq = torch.zeros_like(robot_q)
        q_arm = torch.tensor(np.asarray(self.traj["robot_joint_pos"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
        robot_q[self.arm_joint_ids] = q_arm[: len(self.arm_joint_ids)]
        if "robot_joint_vel" in self.traj:
            dq_arm = torch.tensor(np.asarray(self.traj["robot_joint_vel"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
            robot_dq[self.arm_joint_ids] = dq_arm[: len(self.arm_joint_ids)]
        gripper = torch.tensor(np.asarray(self.traj["gripper_joint_pos"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
        robot_q[self.gripper_joint_ids[0]] = gripper[0]
        if "gripper_joint_vel" in self.traj:
            grip_vel = torch.tensor(np.asarray(self.traj["gripper_joint_vel"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
            robot_dq[self.gripper_joint_ids[0]] = grip_vel[0]
        self.robot.write_joint_state_to_sim(robot_q.unsqueeze(0), robot_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        door_q = self.door.data.joint_pos[0].clone()
        door_dq = torch.zeros_like(door_q)
        door_pos = self._frame_field("door_joint_pos", index)
        handle_pos = self._frame_field("handle_joint_pos", index)
        if door_pos is not None:
            door_q[self.door_joint_id] = float(np.asarray(door_pos).reshape(-1)[0])
        if handle_pos is not None:
            door_q[self.handle_joint_id] = float(np.asarray(handle_pos).reshape(-1)[0])
        door_vel = self._frame_field("door_joint_vel", index)
        handle_vel = self._frame_field("handle_joint_vel", index)
        if door_vel is not None:
            door_dq[self.door_joint_id] = float(np.asarray(door_vel).reshape(-1)[0])
        if handle_vel is not None:
            door_dq[self.handle_joint_id] = float(np.asarray(handle_vel).reshape(-1)[0])
        self.door.write_joint_state_to_sim(door_q.unsqueeze(0), door_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        self._sync_action_targets(q_arm)
        self.env.sim.forward()

    def metrics(self):
        link6_pos = self.robot.data.body_pos_w[0, self.link6_id, :]
        link6_quat = self.robot.data.body_quat_w[0, self.link6_id, :]
        ee_off = torch.tensor((0.1523, 0.0, 0.0), dtype=link6_pos.dtype, device=link6_pos.device).view(1, 3)
        tcp = link6_pos + quat_rotate(link6_quat.view(1, 4), ee_off)[0]

        handle_pos = self.door.data.body_pos_w[0, self.handle_body_id, :]
        handle_quat = self.door.data.body_quat_w[0, self.handle_body_id, :]
        h_off = torch.tensor((-0.08, 0.04, 0.01), dtype=handle_pos.dtype, device=handle_pos.device).view(1, 3)
        target = handle_pos + quat_rotate(handle_quat.view(1, 4), h_off)[0]

        p_left = self.robot.data.body_pos_w[0, self.left_body_id, :]
        p_right = self.robot.data.body_pos_w[0, self.right_body_id, :]
        finger_mid = 0.5 * (p_left + p_right)
        quality = float("nan")
        if self.quality_cfgs is not None:
            try:
                quality = float(compute_stage1_grasp_quality(self.env, **self.quality_cfgs)["quality"][0].item())
            except Exception as exc:
                if not self.warned_quality:
                    print(f"[WARN] grasp_quality failed: {exc}")
                    self.warned_quality = True
        handle_joint = float(self.door.data.joint_pos[0, self.handle_joint_id].item())
        door_joint = float(self.door.data.joint_pos[0, self.door_joint_id].item())
        return {
            "ee_tcp_pos_w": tcp.detach().cpu().numpy().astype(np.float32),
            "handle_grasp_target_w": target.detach().cpu().numpy().astype(np.float32),
            "finger_mid_w": finger_mid.detach().cpu().numpy().astype(np.float32),
            "tcp_to_grasp_dist": float(torch.linalg.norm(tcp - target).item()),
            "finger_mid_to_grasp_dist": float(torch.linalg.norm(finger_mid - target).item()),
            "grasp_quality": quality,
            "door_open": max(door_joint, 0.0),
            "handle_joint_pos": handle_joint,
        }

    def draw_debug(self, metrics: dict):
        if self.draw is None:
            return
        try:
            self.draw.clear_points()
            self.draw.clear_lines()
            tcp = tuple(float(x) for x in metrics["ee_tcp_pos_w"])
            target = tuple(float(x) for x in metrics["handle_grasp_target_w"])
            finger = tuple(float(x) for x in metrics["finger_mid_w"])
            self.preview_tcp_points.append(tcp)
            self.draw.draw_points(
                [tcp, target, finger],
                [(1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.55, 1.0), (0.0, 1.0, 0.0, 1.0)],
                [10.0, 10.0, 8.0],
            )
            if len(self.preview_tcp_points) > 1:
                self.draw.draw_lines(
                    self.preview_tcp_points[:-1],
                    self.preview_tcp_points[1:],
                    [(0.1, 0.25, 1.0, 1.0)] * (len(self.preview_tcp_points) - 1),
                    [2.0] * (len(self.preview_tcp_points) - 1),
                )
            if self.source_tcp is not None and len(self.source_tcp) > 1:
                src = [tuple(float(x) for x in row) for row in self.source_tcp]
                self.draw.draw_lines(src[:-1], src[1:], [(0.45, 0.45, 0.45, 1.0)] * (len(src) - 1), [1.0] * (len(src) - 1))
        except Exception as exc:
            print(f"[WARN] debug draw failed: {exc}")
            self.draw = None


def _print_help():
    print("Controls: Space pause/play | Left frame-1 | Right frame+1 | R restart | Q quit | H help")


def _format_optional(value, precision=4):
    if value is None:
        return "N/A"
    try:
        if not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{precision}f}"
    except Exception:
        return "N/A"


def _replay_trajectory(output_path: str, source_path: str | None):
    _launch_isaac_if_needed()
    traj = _load_npz(output_path)
    length = _trajectory_len(traj, "output_trajectory")
    source_tcp = None
    if source_path is not None:
        try:
            source = _load_npz(source_path)
            if "ee_tcp_pos_w" in source:
                source_tcp = np.asarray(source["ee_tcp_pos_w"], dtype=np.float32)
        except Exception as exc:
            print(f"[WARN] Could not load source TCP path for gray debug line: {exc}")
    elif "merged_from_source" in traj:
        try:
            path = str(np.asarray(traj["merged_from_source"]).item())
            if os.path.exists(path):
                source = _load_npz(path)
                if "ee_tcp_pos_w" in source:
                    source_tcp = np.asarray(source["ee_tcp_pos_w"], dtype=np.float32)
        except Exception:
            source_tcp = None

    start = int(np.clip(args_cli.start_replay_frame, 0, length - 1))
    end = length - 1 if args_cli.end_replay_frame < 0 else int(np.clip(args_cli.end_replay_frame, 0, length - 1))
    if end < start:
        raise ValueError(f"end_replay_frame {end} is smaller than start_replay_frame {start}.")

    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.terminations = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.reset()

    max_jump_after = float(np.asarray(traj.get("max_abs_joint_frame_jump_after", np.asarray([np.nan]))).reshape(-1)[0])
    runtime = FullTrajectoryReplay(base_env, traj, source_tcp, max_jump_after)
    controller = ReplayController(start)
    video = VideoRecorder(_resolve_output_path(args_cli.save_video) if args_cli.save_video else None, args_cli.playback_dt)
    _print_help()

    frame = start
    last_time = 0.0
    last_print_frame = None
    try:
        while simulation_app.is_running() and not controller.quit_requested:
            if controller.help_requested:
                _print_help()
                controller.help_requested = False
            if controller.reset_requested:
                frame = start
                runtime.preview_tcp_points = []
                controller.reset_requested = False
            if controller.step_delta != 0:
                frame = int(np.clip(frame + controller.step_delta, start, end))
                controller.step_delta = 0
            elif not controller.paused and time.time() - last_time >= float(args_cli.playback_dt):
                if frame < end:
                    frame += 1
                elif args_cli.loop:
                    frame = start
                    runtime.preview_tcp_points = []
                else:
                    controller.paused = True
                last_time = time.time()

            runtime.apply_frame(frame)
            base_env.sim.step(render=True)
            base_env.scene.update(base_env.step_dt)
            metrics = runtime.metrics()
            runtime.draw_debug(metrics)

            if last_print_frame != frame:
                last_print_frame = frame
                state = "paused" if controller.paused else "playing"
                print(
                    f"[HUD] frame={frame}/{length - 1} {state} "
                    f"door_open={_format_optional(metrics.get('door_open'))} "
                    f"handle_joint_pos={_format_optional(metrics.get('handle_joint_pos'))} "
                    f"tcp_to_grasp_dist={_format_optional(metrics.get('tcp_to_grasp_dist'))} "
                    f"finger_mid_to_grasp_dist={_format_optional(metrics.get('finger_mid_to_grasp_dist'))} "
                    f"grasp_quality={_format_optional(metrics.get('grasp_quality'))} "
                    f"max_joint_jump_after_merge={_format_optional(runtime.max_jump_after, 6)}"
                )
            video.capture(frame)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        video.save()
        controller.close()
        env.close()
        simulation_app.close()


def main():
    output_path = _resolve_output_path(args_cli.output_trajectory)
    source_path = None

    if args_cli.only_replay:
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"output_trajectory not found for --only_replay: {output_path}")
        print(f"[INFO] only_replay output_trajectory: {output_path}")
        _replay_trajectory(output_path, None)
        return

    if args_cli.source_trajectory is None:
        raise ValueError("--source_trajectory is required unless --only_replay is set.")
    if args_cli.preview_trajectory is None and args_cli.manual_waypoints is None:
        raise ValueError("--preview_trajectory or --manual_waypoints is required unless --only_replay is set.")
    source_path = _resolve_input_path(args_cli.source_trajectory)
    preview_path = _resolve_input_path(args_cli.preview_trajectory) if args_cli.preview_trajectory is not None else None
    print(f"[INFO] source_trajectory resolved to: {source_path}")
    if preview_path is not None:
        print(f"[INFO] preview_trajectory resolved to: {preview_path}")
    print(f"[INFO] output_trajectory resolved to: {output_path}")
    _merge_trajectories(source_path, preview_path, output_path)

    if args_cli.replay:
        _replay_trajectory(output_path, source_path)
    elif simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
