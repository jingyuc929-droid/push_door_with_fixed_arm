"""Interactive waypoint editor for successful trajectory NPZ files."""

import argparse
import glob
import os
import time

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Edit saved ARX5 door-task waypoints in Isaac Sim.")
parser.add_argument("--trajectory_npz", type=str, required=True, help="Path to success trajectory NPZ from play.py.")
parser.add_argument("--output_npz", type=str, required=True, help="Path to save edited manual waypoints NPZ.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Task name.")
parser.add_argument("--env_id", type=int, default=0, help="Environment index to edit. This tool creates one env.")
parser.add_argument(
    "--waypoints",
    nargs="+",
    default=["W2_grasp", "W3_press", "W4_unlock", "W5_open_mid"],
    help="Waypoint names to edit.",
)
parser.add_argument("--print_hz", type=float, default=4.0, help="Terminal HUD update frequency.")
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


WAYPOINT_INDEX_NAMES = ["W1_pregrasp", "W2_grasp", "W3_press", "W4_unlock", "W5_open_mid", "W6_success"]
CTRL_KEYS = {"LEFT_CONTROL", "RIGHT_CONTROL", "CONTROL", "LEFT_CTRL", "RIGHT_CTRL", "CTRL"}
SHIFT_KEYS = {"LEFT_SHIFT", "RIGHT_SHIFT", "SHIFT"}


def _key_name(key) -> str:
    return str(key).split(".")[-1].upper()


def _resolve_input_path(path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(path)) if os.path.isabs(os.path.expanduser(path)) else os.path.expanduser(path)
    candidates = []
    if os.path.isabs(expanded):
        candidates.append(expanded)
    else:
        candidates.append(os.path.abspath(expanded))
        candidates.append(os.path.abspath(os.path.join(PROJECT_ROOT, expanded)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    basename = os.path.basename(path)
    parent = os.path.dirname(path)
    search_dirs = []
    if parent:
        search_dirs.append(os.path.abspath(parent))
        search_dirs.append(os.path.abspath(os.path.join(PROJECT_ROOT, parent)))
    else:
        search_dirs.append(os.getcwd())
        search_dirs.append(PROJECT_ROOT)

    suggestions = []
    for search_dir in search_dirs:
        if os.path.isdir(search_dir):
            exact_env = basename.split("_door", 1)[0] if "_door" in basename else os.path.splitext(basename)[0]
            suggestions.extend(glob.glob(os.path.join(search_dir, f"{exact_env}_door*.npz")))
            suggestions.extend(glob.glob(os.path.join(search_dir, "*.npz")))
    suggestions = sorted(dict.fromkeys(suggestions))[:10]

    message = [f"Input file not found: {path}", "Tried:"]
    message.extend(f"  - {candidate}" for candidate in candidates)
    if suggestions:
        message.append("Nearby candidate files:")
        message.extend(f"  - {candidate}" for candidate in suggestions)
    raise FileNotFoundError("\n".join(message))


def _resolve_output_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(PROJECT_ROOT, expanded))


def _as_1d(data, key: str, index: int, dim: int | None = None):
    if key not in data:
        raise KeyError(f"Trajectory NPZ missing required field '{key}'.")
    arr = np.asarray(data[key])
    value = np.asarray(arr[index]).reshape(-1)
    if dim is not None and value.size != dim:
        raise ValueError(f"Field '{key}' at index {index} has dim {value.size}, expected {dim}.")
    return value.astype(np.float32)


def _find_waypoint_index(data, waypoint_name: str) -> int:
    if "waypoint_indices" not in data:
        raise KeyError("Trajectory NPZ missing 'waypoint_indices'.")
    indices = np.asarray(data["waypoint_indices"]).astype(np.int64).reshape(-1)
    names = WAYPOINT_INDEX_NAMES
    if "waypoint_names" in data:
        names = [str(x) for x in np.asarray(data["waypoint_names"]).reshape(-1)]
    if waypoint_name not in names:
        raise KeyError(f"Waypoint '{waypoint_name}' not found. Available: {names}")
    idx = int(indices[names.index(waypoint_name)])
    if idx < 0:
        raise ValueError(f"Waypoint '{waypoint_name}' has index -1 in source trajectory.")
    return idx


def _load_waypoints(path: str, waypoint_names: list[str]):
    data = np.load(path, allow_pickle=True)
    waypoints = []
    for name in waypoint_names:
        idx = _find_waypoint_index(data, name)
        waypoint = {
            "name": name,
            "source_index": idx,
            "robot_joint_pos": _as_1d(data, "robot_joint_pos", idx, 6),
            "gripper_joint_pos": _as_1d(data, "gripper_joint_pos", idx),
            "door_joint_pos": _as_1d(data, "door_joint_pos", idx)[0],
            "handle_joint_pos": _as_1d(data, "handle_joint_pos", idx)[0],
            "ee_tcp_pos_w": _as_1d(data, "ee_tcp_pos_w", idx, 3),
            "ee_tcp_quat_w": _as_1d(data, "ee_tcp_quat_w", idx, 4) if "ee_tcp_quat_w" in data else np.full(4, np.nan),
        }
        waypoints.append(waypoint)
    return waypoints


def _joint_ids(asset, names: list[str]):
    available = list(asset.data.joint_names)
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError(f"Missing joints {missing}. Available joints: {available}")
    return [available.index(name) for name in names]


def _body_id(asset, name: str):
    available = list(asset.data.body_names)
    if name not in available:
        raise RuntimeError(f"Missing body '{name}'. Available bodies: {available}")
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


class WaypointEditor:
    def __init__(self, env, waypoints: list[dict]):
        self.env = env
        self.robot = env.scene["robot"]
        self.door = env.scene["door"]
        self.waypoints = waypoints
        self.current_wp = 0
        self.selected_joint = 0
        self.quit_requested = False
        self.save_requested = False
        self.last_print = 0.0
        self.warned_quality = False

        self.arm_joint_ids = _joint_ids(self.robot, [f"joint{i}" for i in range(1, 7)])
        self.gripper_joint_ids = _joint_ids(self.robot, ["gripper_joint"])
        self.door_joint_id = _joint_ids(self.door, ["door_joint"])[0]
        self.handle_joint_id = _joint_ids(self.door, ["handle_joint"])[0]

        self.arm_action = _get_action_term(env, "arm_action")
        self.gripper_action = _get_action_term(env, "gripper_action")
        self.quality_cfgs = self._make_quality_cfgs()

        self.original = [
            {
                "robot_joint_pos": wp["robot_joint_pos"].copy(),
                "gripper_joint_pos": wp["gripper_joint_pos"].copy(),
                "door_joint_pos": float(wp["door_joint_pos"]),
                "handle_joint_pos": float(wp["handle_joint_pos"]),
            }
            for wp in waypoints
        ]

        self._subscribe_keyboard()
        self.apply_current_waypoint()

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

    def _subscribe_keyboard(self):
        app_window = omni.appwindow.get_default_app_window()
        self.keyboard = app_window.get_keyboard()
        self.input_iface = carb.input.acquire_input_interface()
        self.keyboard_sub = self.input_iface.subscribe_to_keyboard_events(self.keyboard, self._on_keyboard_event)

    def close(self):
        if getattr(self, "keyboard_sub", None) is not None:
            self.input_iface.unsubscribe_to_keyboard_events(self.keyboard, self.keyboard_sub)
            self.keyboard_sub = None

    @property
    def current(self):
        return self.waypoints[self.current_wp]

    def _shift_pressed(self) -> bool:
        return any(key in self.pressed for key in SHIFT_KEYS) if hasattr(self, "pressed") else False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if not hasattr(self, "pressed"):
            self.pressed = set()
        key = _key_name(event.input)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self.pressed.add(key)
            self._handle_key_press(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key)
        return True

    def _handle_key_press(self, key: str):
        if key in {"KEY_1", "NUM_1", "1"}:
            self.selected_joint = 0
        elif key in {"KEY_2", "NUM_2", "2"}:
            self.selected_joint = 1
        elif key in {"KEY_3", "NUM_3", "3"}:
            self.selected_joint = 2
        elif key in {"KEY_4", "NUM_4", "4"}:
            self.selected_joint = 3
        elif key in {"KEY_5", "NUM_5", "5"}:
            self.selected_joint = 4
        elif key in {"KEY_6", "NUM_6", "6"}:
            self.selected_joint = 5
        elif key in {"UP", "ARROW_UP"}:
            self.adjust_joint(+self._joint_step())
        elif key in {"DOWN", "ARROW_DOWN"}:
            self.adjust_joint(-self._joint_step())
        elif key in {"LEFT_BRACKET", "BRACKETLEFT", "["}:
            self.adjust_gripper(-0.001)
        elif key in {"RIGHT_BRACKET", "BRACKETRIGHT", "]"}:
            self.adjust_gripper(+0.001)
        elif key == "N":
            self.next_waypoint(+1)
        elif key == "P":
            self.next_waypoint(-1)
        elif key == "R":
            self.restore_current()
        elif key == "S":
            self.save_requested = True
        elif key == "Q":
            self.quit_requested = True
        self.apply_current_waypoint()

    def _joint_step(self):
        return 0.05 if self._shift_pressed() else 0.01

    def adjust_joint(self, delta: float):
        self.current["robot_joint_pos"][self.selected_joint] += float(delta)

    def adjust_gripper(self, delta: float):
        self.current["gripper_joint_pos"] = np.asarray(self.current["gripper_joint_pos"], dtype=np.float32)
        self.current["gripper_joint_pos"][0] += float(delta)

    def next_waypoint(self, direction: int):
        self.current_wp = (self.current_wp + int(direction)) % len(self.waypoints)
        print(f"[WAYPOINT] current={self.current['name']} source_index={self.current['source_index']}")

    def restore_current(self):
        orig = self.original[self.current_wp]
        self.current["robot_joint_pos"] = orig["robot_joint_pos"].copy()
        self.current["gripper_joint_pos"] = orig["gripper_joint_pos"].copy()
        self.current["door_joint_pos"] = float(orig["door_joint_pos"])
        self.current["handle_joint_pos"] = float(orig["handle_joint_pos"])
        print(f"[RESTORE] {self.current['name']}")

    def _sync_action_targets(self, q_arm: torch.Tensor):
        if self.arm_action is not None:
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

    def apply_current_waypoint(self):
        wp = self.current
        device = self.env.device

        robot_q = self.robot.data.joint_pos[0].clone()
        robot_dq = torch.zeros_like(robot_q)
        q_arm = torch.tensor(wp["robot_joint_pos"], dtype=robot_q.dtype, device=device)
        robot_q[self.arm_joint_ids] = q_arm
        grip = torch.tensor(wp["gripper_joint_pos"], dtype=robot_q.dtype, device=device).flatten()
        robot_q[self.gripper_joint_ids[0]] = grip[0]
        self.robot.write_joint_state_to_sim(robot_q.unsqueeze(0), robot_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        door_q = self.door.data.joint_pos[0].clone()
        door_dq = torch.zeros_like(door_q)
        door_q[self.door_joint_id] = float(wp["door_joint_pos"])
        door_q[self.handle_joint_id] = float(wp["handle_joint_pos"])
        self.door.write_joint_state_to_sim(door_q.unsqueeze(0), door_dq.unsqueeze(0), env_ids=torch.tensor([0], device=device))

        self._sync_action_targets(q_arm)
        self.env.sim.forward()
        self._update_current_tcp_from_sim()

    def _update_current_tcp_from_sim(self):
        try:
            link6_id = _body_id(self.robot, "link6")
            pos = self.robot.data.body_pos_w[0, link6_id, :]
            quat = self.robot.data.body_quat_w[0, link6_id, :]
            offset = torch.tensor((0.1523, 0.0, 0.0), dtype=pos.dtype, device=pos.device).view(1, 3)
            tcp_pos = pos + quat_rotate(quat.view(1, 4), offset)[0]
            tcp_quat = quat / torch.clamp(torch.linalg.norm(quat), min=1.0e-8)
            self.current["ee_tcp_pos_w"] = tcp_pos.detach().cpu().numpy().astype(np.float32)
            self.current["ee_tcp_quat_w"] = tcp_quat.detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass

    def _quality(self):
        nan = float("nan")
        result = {
            "grasp_quality": nan,
            "tcp_to_grasp_dist": nan,
            "contact_ok": False,
            "closed_no_contact": False,
            "single_finger": False,
        }
        if self.quality_cfgs is None:
            return result
        try:
            terms = compute_stage1_grasp_quality(self.env, **self.quality_cfgs)
            result["grasp_quality"] = float(terms["quality"][0].item())
            result["tcp_to_grasp_dist"] = float(terms["tcp_dist"][0].item())
            result["contact_ok"] = bool(terms["contact_ok"][0].item())
            result["closed_no_contact"] = bool(terms["closed_no_contact"][0].item())
            result["single_finger"] = bool(terms["single_finger"][0].item())
        except Exception as exc:
            if not self.warned_quality:
                print(f"[WARN] grasp quality update failed: {exc}")
                self.warned_quality = True
        return result

    def _finger_mid_to_grasp_dist(self):
        try:
            left_id = _body_id(self.robot, "link7")
            right_id = _body_id(self.robot, "link8")
            handle_id = _body_id(self.door, "handle_1")
            p_left = self.robot.data.body_pos_w[0, left_id, :]
            p_right = self.robot.data.body_pos_w[0, right_id, :]
            p_handle = self.door.data.body_pos_w[0, handle_id, :]
            q_handle = self.door.data.body_quat_w[0, handle_id, :]
            offset = torch.tensor((-0.08, 0.04, 0.01), dtype=p_handle.dtype, device=p_handle.device).view(1, 3)
            target = p_handle + quat_rotate(q_handle.view(1, 4), offset)[0]
            return float(torch.linalg.norm(0.5 * (p_left + p_right) - target).item())
        except Exception:
            return float("nan")

    def print_status(self, force: bool = False):
        now = time.time()
        if not force and now - self.last_print < 1.0 / max(1.0e-3, float(args_cli.print_hz)):
            return
        self.last_print = now
        q = self.current["robot_joint_pos"]
        quality = self._quality()
        finger_mid_dist = self._finger_mid_to_grasp_dist()
        handle_pos = float(self.door.data.joint_pos[0, self.handle_joint_id].item())
        door_pos = float(self.door.data.joint_pos[0, self.door_joint_id].item())
        door_open = max(door_pos, 0.0)
        print(
            f"[HUD] wp={self.current['name']} src_idx={self.current['source_index']} "
            f"joint={self.selected_joint + 1} q={[round(float(x), 4) for x in q]} "
            f"grip={float(self.current['gripper_joint_pos'][0]):.4f} "
            f"tcp_dist={quality['tcp_to_grasp_dist']:.4f} finger_mid_dist={finger_mid_dist:.4f} "
            f"grasp_q={quality['grasp_quality']:.4f} contact_ok={quality['contact_ok']} "
            f"closed_no_contact={quality['closed_no_contact']} single_finger={quality['single_finger']} "
            f"handle={handle_pos:.4f} door_open={door_open:.4f}"
        )

    def save(self, output_path: str, source_path: str):
        old_wp = self.current_wp
        for idx in range(len(self.waypoints)):
            self.current_wp = idx
            self.apply_current_waypoint()
        self.current_wp = old_wp
        self.apply_current_waypoint()

        payload = {
            "source_trajectory_npz": np.asarray(source_path),
            "waypoint_names": np.asarray([wp["name"] for wp in self.waypoints]),
            "source_frame_indices": np.asarray([wp["source_index"] for wp in self.waypoints], dtype=np.int64),
        }
        for wp in self.waypoints:
            short = wp["name"].split("_", 1)[0]
            payload[f"{short}_robot_joint_pos"] = np.asarray(wp["robot_joint_pos"], dtype=np.float32)
            payload[f"{short}_gripper_joint_pos"] = np.asarray(wp["gripper_joint_pos"], dtype=np.float32)
            payload[f"{short}_ee_tcp_pos_w"] = np.asarray(wp["ee_tcp_pos_w"], dtype=np.float32)
            payload[f"{short}_ee_tcp_quat_w"] = np.asarray(wp["ee_tcp_quat_w"], dtype=np.float32)
            payload[f"{short}_handle_joint_pos"] = np.asarray([wp["handle_joint_pos"]], dtype=np.float32)
            payload[f"{short}_door_joint_pos"] = np.asarray([wp["door_joint_pos"]], dtype=np.float32)
            payload[f"{short}_source_frame_index"] = np.asarray([wp["source_index"]], dtype=np.int64)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        np.savez_compressed(output_path, **payload)
        print(f"[SAVE] Manual waypoints saved to: {output_path}")


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


def main():
    trajectory_path = _resolve_input_path(args_cli.trajectory_npz)
    output_path = _resolve_output_path(args_cli.output_npz)
    print(f"[INFO] trajectory_npz resolved to: {trajectory_path}")
    print(f"[INFO] output_npz resolved to: {output_path}")
    waypoints = _load_waypoints(trajectory_path, args_cli.waypoints)
    print("[INFO] Loaded waypoints:")
    for wp in waypoints:
        print(f"  - {wp['name']}: source_index={wp['source_index']}")

    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device
    env_cfg.terminations = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    base_env.reset()
    editor = WaypointEditor(base_env, waypoints)

    print(
        "\nControls: 1-6 select joint | Up/Down +/-0.01 | Shift+Up/Down +/-0.05 | "
        "[/] gripper | N/P waypoint | R restore | S save | Q quit\n"
    )
    try:
        while simulation_app.is_running() and not editor.quit_requested:
            editor.apply_current_waypoint()
            editor.print_status()
            if editor.save_requested:
                editor.save(output_path, trajectory_path)
                editor.save_requested = False
            base_env.sim.step(render=True)
            base_env.scene.update(base_env.step_dt)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        editor.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
