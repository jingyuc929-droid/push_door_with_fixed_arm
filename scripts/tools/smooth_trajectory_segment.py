from __future__ import annotations

"""Smooth a local segment of a full trajectory NPZ and optionally replay it."""

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
        "Smooth robot_joint_pos/gripper_joint_pos in one segment of a full trajectory.\n\n"
        "Examples:\n"
        "  python scripts/tools/smooth_trajectory_segment.py --task Template-Door-Env-v0 "
        "--trajectory_npz logs/trajectories/full_preview_trajectory.npz "
        "--output_npz logs/trajectories/full_smoothed_trajectory.npz "
        "--smooth_start 0 --smooth_end W2 --method min_jerk --blend_radius 10 --replay\n\n"
        "  python scripts/tools/smooth_trajectory_segment.py --task Template-Door-Env-v0 "
        "--trajectory_npz logs/trajectories/full_preview_trajectory.npz "
        "--output_npz logs/trajectories/full_smoothed_trajectory.npz "
        "--smooth_start 0 --smooth_end 185 --method min_jerk --blend_radius 10 --replay\n\n"
        "  python scripts/tools/smooth_trajectory_segment.py "
        "--trajectory_npz logs/trajectories/full_preview_trajectory.npz "
        "--output_npz logs/trajectories/full_smoothed_trajectory_savgol.npz "
        "--smooth_start 0 --smooth_end W2 --method savgol --replay"
    ),
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument("--trajectory_npz", type=str, required=True, help="Input full trajectory NPZ.")
parser.add_argument("--output_npz", type=str, required=True, help="Output full smoothed trajectory NPZ.")
parser.add_argument("--smooth_start", type=str, required=True, help="Start frame or waypoint name W1..W6.")
parser.add_argument("--smooth_end", type=str, required=True, help="End frame or waypoint name W1..W6.")
parser.add_argument("--method", choices=["min_jerk", "cubic_spline", "savgol"], default="min_jerk")
parser.add_argument("--blend_radius", type=int, default=10, help="Raised-cosine blend radius inside segment boundaries.")
parser.add_argument(
    "--smooth_gripper",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Smooth gripper_joint_pos as well as robot_joint_pos.",
)
parser.add_argument(
    "--preserve_shape_weight",
    type=float,
    default=0.0,
    help="0 regenerates from endpoints; >0 mixes in some original segment shape.",
)
parser.add_argument("--replay", action="store_true", default=False, help="Replay the smoothed full trajectory after saving.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Task name.")
parser.add_argument("--playback_dt", type=float, default=0.03, help="Wall-clock delay per replay frame.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop replay.")
parser.add_argument(
    "--draw_debug",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Draw ee_tcp, handle target, finger midpoint, and TCP path during replay.",
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


WAYPOINT_ORDER = ["W1", "W2", "W3", "W4", "W5", "W6"]
WAYPOINT_FULL_NAMES = ["W1_pregrasp", "W2_grasp", "W3_press", "W4_unlock", "W5_open_mid", "W6_success"]


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


def _trajectory_len(data: dict) -> int:
    if "robot_joint_pos" not in data:
        raise KeyError("trajectory_npz missing required field 'robot_joint_pos'.")
    arr = np.asarray(data["robot_joint_pos"])
    if arr.ndim == 0:
        raise ValueError("robot_joint_pos must be time-indexed.")
    return int(arr.shape[0])


def _resolve_frame_token(token: str, data: dict, length: int, arg_name: str) -> tuple[int, str]:
    token = str(token).strip()
    try:
        frame = int(token)
        if frame < 0 or frame >= length:
            raise ValueError(f"{arg_name} frame {frame} out of range 0..{length - 1}.")
        return frame, "explicit frame index"
    except ValueError:
        pass

    waypoint = token.upper()
    if waypoint not in WAYPOINT_ORDER:
        raise ValueError(f"Could not resolve {token}. Please pass --{arg_name} <frame_index>.")

    manifest_key = f"{waypoint}_idx"
    if manifest_key in data:
        frame = int(np.asarray(data[manifest_key]).reshape(-1)[0])
        return _checked_waypoint_frame(frame, length, waypoint, manifest_key), manifest_key

    if "waypoint_indices" in data:
        indices = np.asarray(data["waypoint_indices"], dtype=np.int64).reshape(-1)
        names = WAYPOINT_FULL_NAMES
        if "waypoint_names" in data:
            names = [str(name) for name in np.asarray(data["waypoint_names"]).reshape(-1)]
        short_names = [name.split("_", 1)[0].upper() for name in names]
        if waypoint in short_names:
            pos = short_names.index(waypoint)
            if pos < indices.shape[0]:
                frame = int(indices[pos])
                return _checked_waypoint_frame(frame, length, waypoint, "waypoint_indices"), "waypoint_indices"

    raise ValueError(f"Could not resolve {waypoint}. Please pass --{arg_name} <frame_index>.")


def _checked_waypoint_frame(frame: int, length: int, waypoint: str, source: str) -> int:
    if frame < 0 or frame >= length:
        raise ValueError(f"{waypoint} resolved from {source} to out-of-range frame {frame}; valid range is 0..{length - 1}.")
    return frame


def _min_jerk_segment(arr: np.ndarray, s: int, e: int, preserve_shape_weight: float) -> np.ndarray:
    segment = np.asarray(arr[s : e + 1], dtype=np.float32)
    n = segment.shape[0]
    if n < 2:
        return segment.copy()
    q0 = segment[0]
    q1 = segment[-1]
    tau = np.linspace(0.0, 1.0, n, dtype=np.float32).reshape((-1,) + (1,) * (segment.ndim - 1))
    h = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    smooth = q0 + h * (q1 - q0)
    weight = float(np.clip(preserve_shape_weight, 0.0, 1.0))
    if weight > 0.0:
        smooth = (1.0 - weight) * smooth + weight * segment
        smooth[0] = q0
        smooth[-1] = q1
    return smooth.astype(arr.dtype, copy=False)


def _cubic_spline_segment(arr: np.ndarray, s: int, e: int, preserve_shape_weight: float) -> np.ndarray:
    try:
        from scipy.interpolate import CubicSpline
    except Exception as exc:
        raise RuntimeError("scipy is unavailable for cubic_spline; please use --method min_jerk.") from exc

    segment = np.asarray(arr[s : e + 1], dtype=np.float32)
    n = segment.shape[0]
    if n < 2:
        return segment.copy()
    knot_step = 20
    knot_idx = np.unique(np.r_[0, np.arange(knot_step, n - 1, knot_step), n - 1]).astype(np.int64)
    if float(preserve_shape_weight) <= 0.0:
        knot_idx = np.asarray([0, n - 1], dtype=np.int64)
    x = knot_idx.astype(np.float32)
    y = segment[knot_idx]
    x_new = np.arange(n, dtype=np.float32)
    if len(knot_idx) == 2:
        tau = (x_new / max(n - 1, 1)).reshape((-1,) + (1,) * (segment.ndim - 1))
        smooth = y[0] + tau * (y[-1] - y[0])
    else:
        cs = CubicSpline(x, y, axis=0, bc_type="natural")
        smooth = cs(x_new)
    smooth[0] = segment[0]
    smooth[-1] = segment[-1]
    return np.asarray(smooth, dtype=arr.dtype)


def _savgol_segment(arr: np.ndarray, s: int, e: int) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
    except Exception as exc:
        raise RuntimeError("scipy is unavailable for savgol; please use --method min_jerk.") from exc

    segment = np.asarray(arr[s : e + 1], dtype=np.float32)
    n = segment.shape[0]
    if n < 5:
        return segment.copy()
    window = min(31, n if n % 2 == 1 else n - 1)
    if window < 5:
        return segment.copy()
    polyorder = min(3, window - 2)
    smooth = savgol_filter(segment, window_length=window, polyorder=polyorder, axis=0, mode="interp")
    smooth[0] = segment[0]
    smooth[-1] = segment[-1]
    return np.asarray(smooth, dtype=arr.dtype)


def _smooth_segment(arr: np.ndarray, s: int, e: int, method: str, preserve_shape_weight: float) -> np.ndarray:
    if method == "min_jerk":
        return _min_jerk_segment(arr, s, e, preserve_shape_weight)
    if method == "cubic_spline":
        return _cubic_spline_segment(arr, s, e, preserve_shape_weight)
    if method == "savgol":
        return _savgol_segment(arr, s, e)
    raise ValueError(f"Unsupported smoothing method: {method}")


def _apply_boundary_blend(source_segment: np.ndarray, smooth_segment: np.ndarray, blend_radius: int) -> np.ndarray:
    out = np.asarray(smooth_segment).copy()
    radius = min(max(int(blend_radius), 0), max(0, out.shape[0] - 1) // 2)
    if radius <= 0:
        return out
    for k in range(radius + 1):
        alpha = 0.5 * (1.0 - math.cos(math.pi * k / radius))
        out[k] = (1.0 - alpha) * source_segment[k] + alpha * smooth_segment[k]
    for k in range(radius + 1):
        alpha = 0.5 * (1.0 + math.cos(math.pi * k / radius))
        idx = out.shape[0] - radius - 1 + k
        out[idx] = alpha * smooth_segment[idx] + (1.0 - alpha) * source_segment[idx]
    out[0] = source_segment[0]
    out[-1] = source_segment[-1]
    return out


def _jump_stats(arr: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] < 2:
        return 0.0, 0.0
    jumps = np.abs(np.diff(arr, axis=0))
    per_frame = np.max(jumps.reshape(jumps.shape[0], -1), axis=1)
    return float(np.max(per_frame)), float(np.mean(per_frame))


def _smooth_trajectory(input_path: str, output_path: str) -> tuple[dict, dict]:
    data = _load_npz(input_path)
    length = _trajectory_len(data)
    s, start_source = _resolve_frame_token(args_cli.smooth_start, data, length, "smooth_start")
    e, end_source = _resolve_frame_token(args_cli.smooth_end, data, length, "smooth_end")
    if e <= s:
        raise ValueError(f"smooth_end must be greater than smooth_start, got {s}..{e}.")

    out = {key: np.asarray(value).copy() for key, value in data.items()}
    before_max, before_mean = _jump_stats(data["robot_joint_pos"])
    before_gripper_max, _ = _jump_stats(data["gripper_joint_pos"]) if "gripper_joint_pos" in data else (float("nan"), float("nan"))

    robot_source = np.asarray(data["robot_joint_pos"])
    robot_smooth = _smooth_segment(robot_source, s, e, args_cli.method, args_cli.preserve_shape_weight)
    robot_smooth = _apply_boundary_blend(robot_source[s : e + 1], robot_smooth, args_cli.blend_radius)
    out["robot_joint_pos"][s : e + 1] = robot_smooth

    edited_fields = ["robot_joint_pos"]
    if args_cli.smooth_gripper and "gripper_joint_pos" in data:
        gripper_source = np.asarray(data["gripper_joint_pos"])
        gripper_smooth = _smooth_segment(gripper_source, s, e, args_cli.method, args_cli.preserve_shape_weight)
        gripper_smooth = _apply_boundary_blend(gripper_source[s : e + 1], gripper_smooth, args_cli.blend_radius)
        out["gripper_joint_pos"][s : e + 1] = gripper_smooth
        edited_fields.append("gripper_joint_pos")

    after_max, after_mean = _jump_stats(out["robot_joint_pos"])
    after_gripper_max, _ = _jump_stats(out["gripper_joint_pos"]) if "gripper_joint_pos" in out else (float("nan"), float("nan"))

    out.update(
        {
            "smoothed_from_trajectory": np.asarray(input_path),
            "smooth_start_frame": np.asarray([s], dtype=np.int64),
            "smooth_end_frame": np.asarray([e], dtype=np.int64),
            "smooth_start_resolved_from": np.asarray(start_source),
            "smooth_end_resolved_from": np.asarray(end_source),
            "smooth_method": np.asarray(args_cli.method),
            "smooth_blend_radius": np.asarray([int(args_cli.blend_radius)], dtype=np.int64),
            "smooth_gripper": np.asarray([bool(args_cli.smooth_gripper)]),
            "smooth_preserve_shape_weight": np.asarray([float(args_cli.preserve_shape_weight)], dtype=np.float32),
            "smooth_edited_fields": np.asarray(edited_fields),
            "derived_fields_stale": np.asarray([True]),
            "smooth_max_abs_joint_frame_jump_before": np.asarray([before_max], dtype=np.float32),
            "smooth_mean_abs_joint_frame_jump_before": np.asarray([before_mean], dtype=np.float32),
            "smooth_max_abs_joint_frame_jump_after": np.asarray([after_max], dtype=np.float32),
            "smooth_mean_abs_joint_frame_jump_after": np.asarray([after_mean], dtype=np.float32),
            "smooth_max_abs_gripper_frame_jump_before": np.asarray([before_gripper_max], dtype=np.float32),
            "smooth_max_abs_gripper_frame_jump_after": np.asarray([after_gripper_max], dtype=np.float32),
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(output_path, **out)

    print(f"[SMOOTH] input: {input_path}")
    print(f"[SMOOTH] output: {output_path}")
    print(f"[SMOOTH] segment: {s}..{e} ({e - s + 1} frames)")
    print(f"[SMOOTH] smooth_start resolved from: {start_source}")
    print(f"[SMOOTH] smooth_end resolved from: {end_source}")
    print(f"[SMOOTH] method={args_cli.method} blend_radius={int(args_cli.blend_radius)} smooth_gripper={bool(args_cli.smooth_gripper)}")
    print(f"[SMOOTH] Max joint frame jump before={before_max:.6f} after={after_max:.6f}")
    print(f"[SMOOTH] Mean joint frame jump before={before_mean:.6f} after={after_mean:.6f}")
    print(f"[SMOOTH] Max gripper frame jump before={before_gripper_max:.6f} after={after_gripper_max:.6f}")
    if after_max > before_max:
        print("[WARN] Max joint frame jump after smoothing is larger than before.")
    if after_max > 0.10:
        print("[WARN] Warning: large frame-to-frame joint jump remains.")
    print("[WARN] Derived fields may be stale because joint trajectory was smoothed.")
    return out, {"start": s, "end": e}


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
    def __init__(self):
        self.paused = False
        self.step_delta = 0
        self.reset_requested = False
        self.quit_requested = False
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
            print("Controls: Space pause/play | Left frame-1 | Right frame+1 | R restart | Q quit | H help")
        return True


class VideoRecorder:
    def __init__(self, output_path: str | None, playback_dt: float):
        self.output_path = output_path
        self.playback_dt = playback_dt
        self.frames = []
        self.enabled = output_path is not None
        self.warned = False
        self.tmpdir = tempfile.mkdtemp(prefix="smooth_replay_") if self.enabled else None

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


class TrajectoryReplay:
    def __init__(self, env, traj: dict):
        self.env = env
        self.robot = env.scene["robot"]
        self.door = env.scene["door"]
        self.traj = traj
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
        self.tcp_path = []
        self.warned_quality = False

    @property
    def length(self):
        return _trajectory_len(self.traj)

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

    def _frame_field(self, key: str, index: int, default=None):
        if key not in self.traj:
            return default
        arr = np.asarray(self.traj[key])
        if arr.ndim == 0 or index >= arr.shape[0]:
            return default
        return arr[index]

    def _sync_action_targets(self, q_arm):
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

    def apply_frame(self, index: int):
        device = self.env.device
        robot_q = self.robot.data.joint_pos[0].clone()
        robot_dq = torch.zeros_like(robot_q)
        q_arm = torch.tensor(np.asarray(self.traj["robot_joint_pos"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
        robot_q[self.arm_joint_ids] = q_arm[: len(self.arm_joint_ids)]
        if "robot_joint_vel" in self.traj:
            dq_arm = torch.tensor(np.asarray(self.traj["robot_joint_vel"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
            robot_dq[self.arm_joint_ids] = dq_arm[: len(self.arm_joint_ids)]
        grip = torch.tensor(np.asarray(self.traj["gripper_joint_pos"][index]).reshape(-1), dtype=robot_q.dtype, device=device)
        robot_q[self.gripper_joint_ids[0]] = grip[0]
        self.robot.write_joint_state_to_sim(robot_q.unsqueeze(0), robot_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        door_q = self.door.data.joint_pos[0].clone()
        door_dq = torch.zeros_like(door_q)
        door_pos = self._frame_field("door_joint_pos", index)
        handle_pos = self._frame_field("handle_joint_pos", index)
        if door_pos is not None:
            door_q[self.door_joint_id] = float(np.asarray(door_pos).reshape(-1)[0])
        if handle_pos is not None:
            door_q[self.handle_joint_id] = float(np.asarray(handle_pos).reshape(-1)[0])
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
        return {
            "ee_tcp_pos_w": tcp.detach().cpu().numpy().astype(np.float32),
            "handle_grasp_target_w": target.detach().cpu().numpy().astype(np.float32),
            "finger_mid_w": finger_mid.detach().cpu().numpy().astype(np.float32),
            "tcp_to_grasp_dist": float(torch.linalg.norm(tcp - target).item()),
            "finger_mid_to_grasp_dist": float(torch.linalg.norm(finger_mid - target).item()),
            "grasp_quality": quality,
            "door_open": max(float(self.door.data.joint_pos[0, self.door_joint_id].item()), 0.0),
            "handle_joint_pos": float(self.door.data.joint_pos[0, self.handle_joint_id].item()),
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
            self.tcp_path.append(tcp)
            self.draw.draw_points(
                [tcp, target, finger],
                [(1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.55, 1.0), (0.0, 1.0, 0.0, 1.0)],
                [10.0, 10.0, 8.0],
            )
            if len(self.tcp_path) > 1:
                self.draw.draw_lines(
                    self.tcp_path[:-1],
                    self.tcp_path[1:],
                    [(0.1, 0.25, 1.0, 1.0)] * (len(self.tcp_path) - 1),
                    [2.0] * (len(self.tcp_path) - 1),
                )
        except Exception as exc:
            print(f"[WARN] debug draw failed: {exc}")
            self.draw = None


def _format_optional(value, precision=4):
    try:
        if value is None or not np.isfinite(float(value)):
            return "N/A"
        return f"{float(value):.{precision}f}"
    except Exception:
        return "N/A"


def _replay(output_path: str):
    _launch_isaac_if_needed()
    traj = _load_npz(output_path)
    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.terminations = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.reset()

    runtime = TrajectoryReplay(base_env, traj)
    controller = ReplayController()
    video = VideoRecorder(_resolve_output_path(args_cli.save_video) if args_cli.save_video else None, args_cli.playback_dt)
    print("Controls: Space pause/play | Left frame-1 | Right frame+1 | R restart | Q quit | H help")

    frame = 0
    last_time = 0.0
    last_print_frame = None
    try:
        while simulation_app.is_running() and not controller.quit_requested:
            if controller.reset_requested:
                frame = 0
                runtime.tcp_path = []
                controller.reset_requested = False
            if controller.step_delta != 0:
                frame = int(np.clip(frame + controller.step_delta, 0, runtime.length - 1))
                controller.step_delta = 0
            elif not controller.paused and time.time() - last_time >= float(args_cli.playback_dt):
                if frame < runtime.length - 1:
                    frame += 1
                elif args_cli.loop:
                    frame = 0
                    runtime.tcp_path = []
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
                    f"[HUD] frame={frame}/{runtime.length - 1} {state} "
                    f"door_open={_format_optional(metrics.get('door_open'))} "
                    f"handle_joint_pos={_format_optional(metrics.get('handle_joint_pos'))} "
                    f"tcp_to_grasp_dist={_format_optional(metrics.get('tcp_to_grasp_dist'))} "
                    f"finger_mid_to_grasp_dist={_format_optional(metrics.get('finger_mid_to_grasp_dist'))} "
                    f"grasp_quality={_format_optional(metrics.get('grasp_quality'))}"
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
    input_path = _resolve_input_path(args_cli.trajectory_npz)
    output_path = _resolve_output_path(args_cli.output_npz)
    _smooth_trajectory(input_path, output_path)
    if args_cli.replay:
        _replay(output_path)
    elif simulation_app is not None:
        simulation_app.close()


if __name__ == "__main__":
    main()
