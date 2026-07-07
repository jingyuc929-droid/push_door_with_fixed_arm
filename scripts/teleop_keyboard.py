"""
Keyboard teleoperation for the ARX5 door environment.

The keyboard command is task-space, then mapped to the existing joint-space
environment action with damped least-squares Jacobian IK.

Focus the Isaac Sim viewport before using the keys.
"""

import argparse
import json
import os
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Keyboard teleoperation for the ARX5 door environment.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Name of the task.")
parser.add_argument(
    "--joint-speed",
    type=float,
    default=1.0,
    help="Deprecated alias for --base-action.",
)
parser.add_argument(
    "--z-step",
    type=float,
    default=0.003,
    help="TCP vertical motion per env step in meters.",
)
parser.add_argument(
    "--forward-step",
    type=float,
    default=0.003,
    help="TCP forward/backward motion per env step in meters, along link6 local X.",
)
parser.add_argument(
    "--roll-step",
    type=float,
    default=0.020,
    help="link6 local-X roll motion per env step in radians.",
)
parser.add_argument(
    "--yaw-step",
    type=float,
    default=0.020,
    help="link6 local-Z yaw motion per env step in radians.",
)
parser.add_argument(
    "--base-action",
    type=float,
    default=0.7,
    help="Normalized joint1 action for base yaw. The env clamps this to [-1, 1].",
)
parser.add_argument(
    "--ik-damping",
    type=float,
    default=0.05,
    help="Damping lambda for damped least-squares IK.",
)
parser.add_argument(
    "--save-dir",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "saved_poses"),
    help="Directory for saved TCP pose JSON files.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import carb
import omni.appwindow
import torch
from isaaclab.envs import ManagerBasedRLEnv

from door_env.tasks.manager_based.door_env.door_env_env_cfg import DoorEnvEnvCfg


CTRL_KEYS = {"LEFT_CONTROL", "RIGHT_CONTROL", "CONTROL", "LEFT_CTRL", "RIGHT_CTRL", "CTRL"}
SAVE_KEY = "S"
RESET_KEY = "Z"
QUIT_KEY = "ESCAPE"
OPEN_GRIPPER_KEY = "J"
CLOSE_GRIPPER_KEY = "K"


def _key_name(key) -> str:
    return str(key).split(".")[-1].upper()


class KeyboardTeleop:
    """Small stateful keyboard device backed by carb input events."""

    def __init__(self, save_callback, reset_callback):
        self._pressed = set()
        self._save_callback = save_callback
        self._reset_callback = reset_callback
        self._quit_requested = False
        self._save_latched = False
        self._reset_latched = False
        self._gripper_command = 1.0

        app_window = omni.appwindow.get_default_app_window()
        self._keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def _ctrl_is_pressed(self) -> bool:
        return any(key in self._pressed for key in CTRL_KEYS)

    def _on_keyboard_event(self, event, *args, **kwargs):
        key = _key_name(event.input)

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(key)
            if key == QUIT_KEY:
                self._quit_requested = True
            elif key == OPEN_GRIPPER_KEY:
                self._gripper_command = 1.0
            elif key == CLOSE_GRIPPER_KEY:
                self._gripper_command = -1.0
            elif key == SAVE_KEY and self._ctrl_is_pressed() and not self._save_latched:
                self._save_latched = True
                self._save_callback()
            elif key == RESET_KEY and self._ctrl_is_pressed() and not self._reset_latched:
                self._reset_latched = True
                self._reset_callback()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(key)
            if key == SAVE_KEY:
                self._save_latched = False
            elif key == RESET_KEY:
                self._reset_latched = False

        return True

    def command_request(self) -> dict[str, float]:
        if self._ctrl_is_pressed():
            return {
                "z": 0.0,
                "forward": 0.0,
                "base": 0.0,
                "roll": 0.0,
                "yaw": 0.0,
                "gripper": self._gripper_command,
            }

        z = 0.0
        forward = 0.0
        base = 0.0
        roll = 0.0
        yaw = 0.0
        if "W" in self._pressed:
            z += 1.0
        if "S" in self._pressed:
            z -= 1.0
        if "I" in self._pressed:
            forward += 1.0
        if "O" in self._pressed:
            forward -= 1.0
        if "A" in self._pressed:
            base += 1.0
        if "D" in self._pressed:
            base -= 1.0
        if "R" in self._pressed:
            roll += 1.0
        if "Q" in self._pressed:
            roll -= 1.0
        if "N" in self._pressed:
            yaw += 1.0
        if "M" in self._pressed:
            yaw -= 1.0
        return {"z": z, "forward": forward, "base": base, "roll": roll, "yaw": yaw, "gripper": self._gripper_command}


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


class JacobianIKTeleop:
    """Maps sparse task-space keyboard requests to the 7D environment action."""

    def __init__(self, env, ee_body_name: str = "link6", tcp_offset_pos=(0.1523, 0.0, 0.0)):
        self.env = env
        self.robot = env.scene["robot"]
        self.ee_body_id = _find_body_id(self.robot, ee_body_name)
        self.tcp_offset_pos = tcp_offset_pos
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.arm_joint_ids = self._find_joint_ids(self.arm_joint_names)
        self.ik_column_ids = list(range(1, 6))
        self.arm_action_term = _get_action_term(env, "arm_action")
        self.delta_scale = self._read_delta_scale()

    def _find_joint_ids(self, joint_names: list[str]) -> list[int]:
        available = list(self.robot.data.joint_names)
        missing = [name for name in joint_names if name not in available]
        if missing:
            raise RuntimeError(f"Missing robot joints {missing}. Available joints: {available}")
        return [available.index(name) for name in joint_names]

    def _read_delta_scale(self) -> torch.Tensor:
        if self.arm_action_term is not None and hasattr(self.arm_action_term, "_delta_scale"):
            scale = self.arm_action_term._delta_scale[0, :6].detach().clone()
        else:
            scale = torch.tensor((0.015, 0.015, 0.015, 0.020, 0.020, 0.020), device=self.env.device)
        return scale.to(device=self.env.device, dtype=torch.float32).clamp_min(1.0e-6)

    def _ee_jacobian(self) -> torch.Tensor:
        jacobians = self.robot.root_physx_view.get_jacobians()
        if jacobians.shape[1] == len(self.robot.data.body_names) - 1:
            jacobian_body_id = self.ee_body_id - 1
        else:
            jacobian_body_id = self.ee_body_id
        if jacobian_body_id < 0 or jacobian_body_id >= jacobians.shape[1]:
            raise RuntimeError(
                f"Invalid Jacobian body index {jacobian_body_id}. "
                f"body_id={self.ee_body_id}, jacobian_shape={tuple(jacobians.shape)}"
            )
        return jacobians[0, jacobian_body_id, :, self.arm_joint_ids].to(device=self.env.device)

    def _tcp_jacobian(self) -> torch.Tensor:
        jacobian = self._ee_jacobian()
        quat = self.robot.data.body_quat_w[0, self.ee_body_id, :]
        offset_b = torch.tensor(self.tcp_offset_pos, dtype=quat.dtype, device=quat.device)
        offset_w = _quat_rotate(quat.unsqueeze(0), offset_b.unsqueeze(0))[0]
        skew_offset = torch.zeros((3, 3), dtype=jacobian.dtype, device=jacobian.device)
        skew_offset[0, 1] = -offset_w[2]
        skew_offset[0, 2] = offset_w[1]
        skew_offset[1, 0] = offset_w[2]
        skew_offset[1, 2] = -offset_w[0]
        skew_offset[2, 0] = -offset_w[1]
        skew_offset[2, 1] = offset_w[0]
        tcp_jacobian = jacobian.clone()
        tcp_jacobian[:3, :] = jacobian[:3, :] - skew_offset @ jacobian[3:6, :]
        return tcp_jacobian

    def _link6_x_axis_w(self) -> torch.Tensor:
        quat = self.robot.data.body_quat_w[0, self.ee_body_id, :]
        axis = torch.tensor((1.0, 0.0, 0.0), dtype=quat.dtype, device=quat.device)
        return _quat_rotate(quat.unsqueeze(0), axis.unsqueeze(0))[0]

    def _link6_z_axis_w(self) -> torch.Tensor:
        quat = self.robot.data.body_quat_w[0, self.ee_body_id, :]
        axis = torch.tensor((0.0, 0.0, 1.0), dtype=quat.dtype, device=quat.device)
        return _quat_rotate(quat.unsqueeze(0), axis.unsqueeze(0))[0]

    def _solve_dls(self, task_jacobian: torch.Tensor, task_delta: torch.Tensor) -> torch.Tensor:
        if task_jacobian.numel() == 0:
            return torch.zeros(len(self.ik_column_ids), dtype=torch.float32, device=self.env.device)
        eye = torch.eye(task_jacobian.shape[0], dtype=torch.float32, device=self.env.device)
        lhs = task_jacobian @ task_jacobian.transpose(0, 1) + float(args_cli.ik_damping) ** 2 * eye
        return task_jacobian.transpose(0, 1) @ torch.linalg.solve(lhs, task_delta)

    def action_from_request(self, request: dict[str, float]) -> torch.Tensor:
        action = torch.zeros(7, dtype=torch.float32, device=self.env.device)
        task_rows = []
        task_delta = []

        z_cmd = float(request["z"])
        forward_cmd = float(request["forward"])
        roll_cmd = float(request["roll"])
        yaw_cmd = float(request["yaw"])
        jacobian = self._tcp_jacobian()[:, self.ik_column_ids]

        if z_cmd != 0.0:
            task_rows.append(jacobian[2:3, :])
            task_delta.append(torch.tensor([z_cmd * float(args_cli.z_step)], device=self.env.device))

        if forward_cmd != 0.0:
            forward_axis_w = self._link6_x_axis_w()
            task_rows.append(jacobian[:3, :])
            task_delta.append(forward_axis_w * forward_cmd * float(args_cli.forward_step))

        if roll_cmd != 0.0:
            roll_axis_w = self._link6_x_axis_w()
            task_rows.append(roll_axis_w.view(1, 3) @ jacobian[3:6, :])
            task_delta.append(torch.tensor([roll_cmd * float(args_cli.roll_step)], device=self.env.device))

        if yaw_cmd != 0.0:
            yaw_axis_w = self._link6_z_axis_w()
            task_rows.append(yaw_axis_w.view(1, 3) @ jacobian[3:6, :])
            task_delta.append(torch.tensor([yaw_cmd * float(args_cli.yaw_step)], device=self.env.device))

        if task_rows:
            j_task = torch.cat(task_rows, dim=0).to(dtype=torch.float32)
            dx_task = torch.cat(task_delta).to(dtype=torch.float32)
            q_delta = self._solve_dls(j_task, dx_task)
            action[1:6] += q_delta / self.delta_scale[1:6]

        base_cmd = float(request["base"])
        if base_cmd != 0.0:
            base_action = float(args_cli.base_action if args_cli.base_action is not None else args_cli.joint_speed)
            action[0] += base_cmd * base_action

        action[:6].clamp_(-1.0, 1.0)
        action[6] = float(request["gripper"])
        return action.unsqueeze(0)


def _find_body_id(robot, body_name: str) -> int:
    body_names = list(robot.data.body_names)
    if body_name not in body_names:
        raise RuntimeError(f"Body '{body_name}' not found. Available bodies: {body_names}")
    return body_names.index(body_name)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(v[..., :1])
    qv = torch.cat((zeros, v), dim=-1)
    qc = torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)
    return _quat_mul(_quat_mul(q, qv), qc)[..., 1:]


def _tcp_pose_w(env, body_id: int, ee_offset_pos=(0.1523, 0.0, 0.0)):
    robot = env.scene["robot"]
    body_pos = robot.data.body_pos_w[0, body_id, :]
    body_quat = robot.data.body_quat_w[0, body_id, :]
    offset = torch.tensor(ee_offset_pos, dtype=body_pos.dtype, device=body_pos.device)
    tcp_pos = body_pos + _quat_rotate(body_quat.unsqueeze(0), offset.unsqueeze(0))[0]
    tcp_quat = body_quat / torch.norm(body_quat).clamp_min(1.0e-8)
    return tcp_pos.detach().cpu(), tcp_quat.detach().cpu()


def _joint_snapshot(env):
    robot = env.scene["robot"]
    names = list(robot.data.joint_names)
    values = robot.data.joint_pos[0].detach().cpu().tolist()
    return {name: float(values[idx]) for idx, name in enumerate(names)}


def _print_controls():
    print(
        """
Keyboard controls
-----------------
W / S : move TCP up / down in world Z
I / O : move TCP forward / backward along link6 local X
A / D : rotate arm left / right around base Z
R / Q : roll link6 clockwise / counterclockwise around local X
N / M : yaw link6 left / right around local Z
J     : open gripper
K     : close gripper
Ctrl+S: save current TCP pose as JSON
Ctrl+Z: reset environment manually
Esc   : quit
Ctrl+C: force quit from terminal

Saved pose format:
  position is [x, y, z] in world frame, meters
  quaternion_wxyz is [w, x, y, z] in world frame
"""
    )


def main():
    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"
    env_cfg.terminations = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot = env.scene["robot"]
    link6_id = _find_body_id(robot, "link6")
    ik_controller = JacobianIKTeleop(env, ee_body_name="link6", tcp_offset_pos=(0.1523, 0.0, 0.0))
    os.makedirs(args_cli.save_dir, exist_ok=True)

    def save_pose():
        pos, quat = _tcp_pose_w(env, link6_id)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(args_cli.save_dir, f"arx5_tcp_pose_{stamp}.json")
        payload = {
            "timestamp": stamp,
            "frame": "world",
            "tcp_body": "link6",
            "tcp_offset_pos": [0.1523, 0.0, 0.0],
            "position": [float(x) for x in pos.tolist()],
            "quaternion_wxyz": [float(x) for x in quat.tolist()],
            "joint_position": _joint_snapshot(env),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        print(f"[SAVE] TCP pose saved to: {path}")
        print(f"       position={payload['position']}")
        print(f"       quaternion_wxyz={payload['quaternion_wxyz']}")

    def reset_env():
        env.reset()
        print("[RESET] Environment reset by Ctrl+Z.")

    keyboard = KeyboardTeleop(save_pose, reset_env)
    _print_controls()

    try:
        while simulation_app.is_running() and not keyboard.quit_requested:
            with torch.inference_mode():
                request = keyboard.command_request()
                actions = ik_controller.action_from_request(request)
                env.step(actions)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        keyboard.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
