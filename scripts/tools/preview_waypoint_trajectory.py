"""Preview interpolated manual waypoint trajectories in Isaac Sim."""

import argparse
import glob
import os
import tempfile
import time

from isaaclab.app import AppLauncher


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected True/False, got {value!r}")


parser = argparse.ArgumentParser(description="Preview manual waypoint trajectory interpolation in Isaac Sim.")
parser.add_argument("--manual_waypoints", type=str, required=True, help="Path to manual_waypoints_v1.npz.")
parser.add_argument("--source_trajectory", type=str, default=None, help="Optional original success trajectory NPZ.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Task name.")
parser.add_argument("--num_interp_per_segment", type=int, default=50, help="Interpolated frames per waypoint segment.")
parser.add_argument("--playback_dt", type=float, default=0.03, help="Playback wall-clock delay per frame.")
parser.add_argument("--draw_original", type=_str_to_bool, default=True, help="Draw source trajectory TCP path if available.")
parser.add_argument("--output_video", type=str, default=None, help="Optional output video path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import numpy as np
import omni.appwindow
import torch
from isaaclab.managers import SceneEntityCfg

import door_env.tasks  # noqa: F401
from door_env.tasks.manager_based.door_env.door_env_env_cfg import DoorEnvEnvCfg
from door_env.tasks.manager_based.door_env.mdp.rewards import compute_stage1_grasp_quality


WAYPOINT_ORDER = ["W2", "W3", "W4", "W5", "W6"]


def _key_name(key) -> str:
    return str(key).split(".")[-1].upper()


def _resolve_input_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    candidates = [expanded] if os.path.isabs(expanded) else [os.path.abspath(expanded), os.path.join(PROJECT_ROOT, expanded)]
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if os.path.exists(candidate):
            return candidate
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    search_dirs = [os.path.abspath(parent), os.path.join(PROJECT_ROOT, parent)] if parent else [os.getcwd(), PROJECT_ROOT]
    suggestions = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            suggestions.extend(glob.glob(os.path.join(search_dir, "*.npz")))
    suggestions = sorted(dict.fromkeys(suggestions))[:10]
    msg = [f"Input file not found: {path}", "Tried:"]
    msg.extend(f"  - {os.path.abspath(c)}" for c in candidates)
    if suggestions:
        msg.append("Nearby candidate files:")
        msg.extend(f"  - {item}" for item in suggestions)
    raise FileNotFoundError("\n".join(msg))


def _resolve_output_path(path: str | None) -> str | None:
    if path is None:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(PROJECT_ROOT, expanded))


def _load_manual_waypoints(path: str):
    data = np.load(path, allow_pickle=True)
    waypoints = []
    for prefix in WAYPOINT_ORDER:
        rq = f"{prefix}_robot_joint_pos"
        gq = f"{prefix}_gripper_joint_pos"
        if rq not in data or gq not in data:
            print(f"[WARN] Manual waypoint {prefix} missing; skipping.")
            continue
        wp = {
            "prefix": prefix,
            "robot_joint_pos": np.asarray(data[rq], dtype=np.float32).reshape(6),
            "gripper_joint_pos": np.asarray(data[gq], dtype=np.float32).reshape(-1),
            "ee_tcp_pos_w": np.asarray(data[f"{prefix}_ee_tcp_pos_w"], dtype=np.float32).reshape(3)
            if f"{prefix}_ee_tcp_pos_w" in data
            else np.full(3, np.nan, dtype=np.float32),
            "door_joint_pos": float(np.asarray(data[f"{prefix}_door_joint_pos"]).reshape(-1)[0])
            if f"{prefix}_door_joint_pos" in data
            else 0.0,
            "handle_joint_pos": float(np.asarray(data[f"{prefix}_handle_joint_pos"]).reshape(-1)[0])
            if f"{prefix}_handle_joint_pos" in data
            else 0.0,
        }
        waypoints.append(wp)
    if len(waypoints) < 2:
        raise RuntimeError("Need at least two manual waypoints to preview a trajectory.")
    return waypoints


def _catmull_rom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
    )


def _interpolate(values: np.ndarray, frames_per_segment: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n = values.shape[0]
    if n < 2:
        return values.copy()
    frames_per_segment = max(1, int(frames_per_segment))
    output = []
    for i in range(n - 1):
        p1 = values[i]
        p2 = values[i + 1]
        p0 = values[max(i - 1, 0)]
        p3 = values[min(i + 2, n - 1)]
        for j in range(frames_per_segment):
            t = j / float(frames_per_segment)
            if n >= 3:
                output.append(_catmull_rom(p0, p1, p2, p3, t))
            else:
                output.append((1.0 - t) * p1 + t * p2)
    output.append(values[-1])
    return np.asarray(output, dtype=np.float32)


def _build_preview_trajectory(waypoints: list[dict], frames_per_segment: int):
    robot = np.stack([wp["robot_joint_pos"] for wp in waypoints], axis=0)
    gripper = np.stack([wp["gripper_joint_pos"] for wp in waypoints], axis=0)
    door = np.asarray([[wp["door_joint_pos"]] for wp in waypoints], dtype=np.float32)
    handle = np.asarray([[wp["handle_joint_pos"]] for wp in waypoints], dtype=np.float32)
    tcp_waypoints = np.stack([wp["ee_tcp_pos_w"] for wp in waypoints], axis=0)
    return {
        "robot_joint_pos": _interpolate(robot, frames_per_segment),
        "gripper_joint_pos": _interpolate(gripper, frames_per_segment),
        "door_joint_pos": _interpolate(door, frames_per_segment).reshape(-1),
        "handle_joint_pos": _interpolate(handle, frames_per_segment).reshape(-1),
        "waypoint_tcp_pos_w": tcp_waypoints,
    }


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


class PreviewController:
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
        if self.sub is not None:
            self.input_iface.unsubscribe_to_keyboard_events(self.keyboard, self.sub)
            self.sub = None

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        key = _key_name(event.input)
        if key == "SPACE":
            self.paused = not self.paused
            print(f"[PREVIEW] paused={self.paused}")
        elif key in {"LEFT", "ARROW_LEFT"}:
            self.step_delta = -1
            self.paused = True
        elif key in {"RIGHT", "ARROW_RIGHT"}:
            self.step_delta = +1
            self.paused = True
        elif key == "R":
            self.reset_requested = True
        elif key == "Q":
            self.quit_requested = True
        return True


class PreviewRuntime:
    def __init__(self, env, trajectory: dict, source_tcp: np.ndarray | None):
        self.env = env
        self.robot = env.scene["robot"]
        self.door = env.scene["door"]
        self.traj = trajectory
        self.source_tcp = source_tcp
        self.arm_joint_ids = _joint_ids(self.robot, [f"joint{i}" for i in range(1, 7)])
        self.gripper_joint_ids = _joint_ids(self.robot, ["gripper_joint"])
        self.door_joint_id = _joint_ids(self.door, ["door_joint"])[0]
        self.handle_joint_id = _joint_ids(self.door, ["handle_joint"])[0]
        self.link6_id = _body_id(self.robot, "link6")
        self.handle_body_id = _body_id(self.door, "handle_1")
        self.left_body_id = _body_id(self.robot, "link7")
        self.right_body_id = _body_id(self.robot, "link8")
        self.quality_cfgs = self._make_quality_cfgs()
        self.draw = self._make_draw()
        self.preview_tcp_points = []
        self.warned_quality = False

    @property
    def length(self):
        return int(self.traj["robot_joint_pos"].shape[0])

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

    def apply_frame(self, index: int):
        index = int(np.clip(index, 0, self.length - 1))
        device = self.env.device
        robot_q = self.robot.data.joint_pos[0].clone()
        robot_dq = torch.zeros_like(robot_q)
        robot_q[self.arm_joint_ids] = torch.tensor(self.traj["robot_joint_pos"][index], dtype=robot_q.dtype, device=device)
        gripper = torch.tensor(self.traj["gripper_joint_pos"][index], dtype=robot_q.dtype, device=device).flatten()
        robot_q[self.gripper_joint_ids[0]] = gripper[0]
        self.robot.write_joint_state_to_sim(robot_q.unsqueeze(0), robot_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        door_q = self.door.data.joint_pos[0].clone()
        door_dq = torch.zeros_like(door_q)
        door_q[self.door_joint_id] = float(self.traj["door_joint_pos"][index])
        door_q[self.handle_joint_id] = float(self.traj["handle_joint_pos"][index])
        self.door.write_joint_state_to_sim(door_q.unsqueeze(0), door_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))
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
        tcp_dist = float(torch.linalg.norm(tcp - target).item())
        finger_mid_dist = float(torch.linalg.norm(finger_mid - target).item())

        quality = float("nan")
        if self.quality_cfgs is not None:
            try:
                terms = compute_stage1_grasp_quality(self.env, **self.quality_cfgs)
                quality = float(terms["quality"][0].item())
            except Exception as exc:
                if not self.warned_quality:
                    print(f"[WARN] grasp_quality failed: {exc}")
                    self.warned_quality = True

        handle_joint = float(self.door.data.joint_pos[0, self.handle_joint_id].item())
        door_open = max(float(self.door.data.joint_pos[0, self.door_joint_id].item()), 0.0)
        return {
            "ee_tcp_pos_w": tcp.detach().cpu().numpy().astype(np.float32),
            "handle_grasp_target_w": target.detach().cpu().numpy().astype(np.float32),
            "finger_mid_w": finger_mid.detach().cpu().numpy().astype(np.float32),
            "tcp_to_grasp_dist": tcp_dist,
            "finger_mid_to_grasp_dist": finger_mid_dist,
            "grasp_quality": quality,
            "door_open": door_open,
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
            finger_mid = tuple(float(x) for x in metrics["finger_mid_w"])
            self.preview_tcp_points.append(tcp)
            self.draw.draw_points(
                [tcp, target, finger_mid],
                [(1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 0.55, 1.0), (0.0, 1.0, 0.0, 1.0)],
                [10.0, 10.0, 8.0],
            )
            if len(self.preview_tcp_points) > 1:
                starts = self.preview_tcp_points[:-1]
                ends = self.preview_tcp_points[1:]
                self.draw.draw_lines(starts, ends, [(0.1, 0.25, 1.0, 1.0)] * len(starts), [2.0] * len(starts))
            if self.source_tcp is not None and args_cli.draw_original and len(self.source_tcp) > 1:
                src = [tuple(float(x) for x in row) for row in self.source_tcp]
                self.draw.draw_lines(src[:-1], src[1:], [(0.45, 0.45, 0.45, 1.0)] * (len(src) - 1), [1.0] * (len(src) - 1))
        except Exception as exc:
            print(f"[WARN] debug draw failed: {exc}")
            self.draw = None


class VideoRecorder:
    def __init__(self, output_path: str | None):
        self.output_path = output_path
        self.frames = []
        self.enabled = output_path is not None
        self.warned = False
        self.tmpdir = None
        if self.enabled:
            self.tmpdir = tempfile.mkdtemp(prefix="waypoint_preview_")

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
                print(f"[WARN] output_video capture unavailable; continuing without video: {exc}")
                self.warned = True
            self.enabled = False

    def save(self):
        if not self.enabled or not self.frames:
            return
        try:
            import imageio.v2 as imageio

            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            with imageio.get_writer(self.output_path, fps=max(1, int(round(1.0 / max(args_cli.playback_dt, 1.0e-6))))) as writer:
                for path in self.frames:
                    if os.path.exists(path):
                        writer.append_data(imageio.imread(path))
            print(f"[VIDEO] saved to: {self.output_path}")
        except Exception as exc:
            print(f"[WARN] Could not save output_video: {exc}")


def _load_source_tcp(path: str | None):
    if path is None:
        return None
    data = np.load(path, allow_pickle=True)
    if "ee_tcp_pos_w" not in data:
        print("[WARN] source_trajectory has no ee_tcp_pos_w; gray original path disabled.")
        return None
    return np.asarray(data["ee_tcp_pos_w"], dtype=np.float32)


def _save_preview_npz(path: str, records: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, **{key: np.asarray(value) for key, value in records.items()})
    print(f"[PREVIEW] saved interpolated trajectory to: {path}")


def main():
    manual_path = _resolve_input_path(args_cli.manual_waypoints)
    source_path = _resolve_input_path(args_cli.source_trajectory) if args_cli.source_trajectory else None
    video_path = _resolve_output_path(args_cli.output_video)
    preview_npz = os.path.join(os.path.dirname(manual_path), "preview_trajectory.npz")
    print(f"[INFO] manual_waypoints resolved to: {manual_path}")
    if source_path:
        print(f"[INFO] source_trajectory resolved to: {source_path}")

    waypoints = _load_manual_waypoints(manual_path)
    trajectory = _build_preview_trajectory(waypoints, args_cli.num_interp_per_segment)
    source_tcp = _load_source_tcp(source_path)

    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.terminations = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.reset()

    runtime = PreviewRuntime(base_env, trajectory, source_tcp)
    controller = PreviewController()
    video = VideoRecorder(video_path)
    records = {
        "robot_joint_pos": trajectory["robot_joint_pos"],
        "gripper_joint_pos": trajectory["gripper_joint_pos"],
        "ee_tcp_pos_w": np.full((trajectory["robot_joint_pos"].shape[0], 3), np.nan, dtype=np.float32),
        "tcp_to_grasp_dist": np.full((trajectory["robot_joint_pos"].shape[0],), np.nan, dtype=np.float32),
        "grasp_quality": np.full((trajectory["robot_joint_pos"].shape[0],), np.nan, dtype=np.float32),
    }

    print("Controls: Space pause/play | Left/Right frame step | R restart | Q quit")
    index = 0
    last_time = 0.0
    try:
        while simulation_app.is_running() and not controller.quit_requested:
            if controller.reset_requested:
                index = 0
                runtime.preview_tcp_points = []
                controller.reset_requested = False
            if controller.step_delta != 0:
                index = int(np.clip(index + controller.step_delta, 0, runtime.length - 1))
                controller.step_delta = 0
            elif not controller.paused and time.time() - last_time >= float(args_cli.playback_dt):
                index = (index + 1) % runtime.length
                last_time = time.time()

            runtime.apply_frame(index)
            base_env.sim.step(render=True)
            base_env.scene.update(base_env.step_dt)
            metrics = runtime.metrics()
            runtime.draw_debug(metrics)
            records["ee_tcp_pos_w"][index] = metrics["ee_tcp_pos_w"]
            records["tcp_to_grasp_dist"][index] = metrics["tcp_to_grasp_dist"]
            records["grasp_quality"][index] = metrics["grasp_quality"]
            if index % 10 == 0:
                print(
                    f"[PREVIEW] frame={index}/{runtime.length - 1} "
                    f"tcp_dist={metrics['tcp_to_grasp_dist']:.4f} "
                    f"finger_mid_dist={metrics['finger_mid_to_grasp_dist']:.4f} "
                    f"grasp_quality={metrics['grasp_quality']:.4f} "
                    f"handle={metrics['handle_joint_pos']:.4f} door_open={metrics['door_open']:.4f}"
                )
            video.capture(index)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        missing = np.nonzero(~np.isfinite(records["tcp_to_grasp_dist"]))[0]
        for fill_idx in missing:
            runtime.apply_frame(fill_idx)
            base_env.sim.forward()
            metrics = runtime.metrics()
            records["ee_tcp_pos_w"][fill_idx] = metrics["ee_tcp_pos_w"]
            records["tcp_to_grasp_dist"][fill_idx] = metrics["tcp_to_grasp_dist"]
            records["grasp_quality"][fill_idx] = metrics["grasp_quality"]
        _save_preview_npz(preview_npz, records)
        video.save()
        controller.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
