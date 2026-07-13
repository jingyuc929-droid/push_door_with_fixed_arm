# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Visual smoke test for DoorBot base target following and arm target tracking."""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Debug target-reaching behavior for the push-door DoorBot task.")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Gym task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments. This debug script expects 1.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--base_target", type=float, nargs=2, default=(0.6, 0.0), metavar=("X", "Y"))
parser.add_argument("--base_target_yaw", type=float, default=-math.pi / 2.0, help="Target base yaw in world frame.")
parser.add_argument("--ee_target", type=float, nargs=3, default=(0.55, 0.05, 0.85), metavar=("X", "Y", "Z"))
parser.add_argument(
    "--arm_q_target",
    type=float,
    nargs=6,
    default=(0.35, 0.75, -0.85, 0.25, 0.35, 0.0),
    metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"),
    help="Joint-space arm target used to validate the arm position-target action path.",
)
parser.add_argument("--max_steps", type=int, default=3000, help="Stop after this many env steps. Use <=0 for infinite.")
parser.add_argument("--print_every", type=int, default=120, help="Print env0 debug metrics every N steps.")
parser.add_argument("--base_kp", type=float, default=0.8, help="Base XY proportional gain.")
parser.add_argument("--yaw_kp", type=float, default=1.2, help="Base yaw proportional gain.")
parser.add_argument("--stop_radius", type=float, default=0.08, help="Zero base velocity inside this XY radius.")
parser.add_argument("--max_vx", type=float, default=0.35, help="Max commanded base vx.")
parser.add_argument("--max_vy", type=float, default=0.20, help="Max commanded base vy.")
parser.add_argument("--max_wz", type=float, default=0.45, help="Max commanded base wz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

import door_env.tasks  # noqa: F401


def _make_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw

        return _debug_draw.acquire_debug_draw_interface()
    except Exception as exc:
        print(f"[WARN] debug_draw unavailable: {exc}")
        return None


def _get_action_term(env):
    return env.unwrapped.action_manager.get_term("high_level_action")


def _get_body_pos_w(robot, body_name: str) -> torch.Tensor | None:
    if body_name not in robot.body_names:
        return None
    body_id = robot.body_names.index(body_name)
    return robot.data.body_pos_w[:, body_id, :]


def _normalize_base_action(term, desired_cmd: torch.Tensor) -> torch.Tensor:
    base_action = (desired_cmd - term._base_mid) / torch.clamp(term._base_half_range, min=1.0e-6)
    return torch.clamp(base_action, -1.0, 1.0)


def _normalize_arm_action(term, q_target: torch.Tensor) -> torch.Tensor:
    default_q = term._asset.data.default_joint_pos[:, term.joint_ids]
    arm_scale = torch.clamp(term._arm_scale, min=1.0e-6)
    arm_action = (q_target - default_q) / arm_scale
    return torch.clamp(arm_action, -1.0, 1.0)


def _build_debug_action(env, term, base_target_xy: torch.Tensor, target_yaw: float, arm_q_target: torch.Tensor):
    robot = env.unwrapped.scene["robot"]
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    target_w = torch.zeros((num_envs, 3), device=device)
    target_w[:, :2] = base_target_xy.view(1, 2)
    target_w[:, 2] = root_pos[:, 2]

    target_vec_w = target_w - root_pos
    target_vec_b = quat_apply_inverse(yaw_quat(root_quat), target_vec_w)
    xy_dist = torch.linalg.norm(target_vec_w[:, :2], dim=-1)

    desired_cmd = torch.zeros((num_envs, 5), device=device)
    desired_cmd[:, 0] = torch.clamp(args_cli.base_kp * target_vec_b[:, 0], -args_cli.max_vx, args_cli.max_vx)
    desired_cmd[:, 1] = torch.clamp(args_cli.base_kp * target_vec_b[:, 1], -args_cli.max_vy, args_cli.max_vy)

    _, _, yaw = euler_xyz_from_quat(root_quat)
    yaw_error = wrap_to_pi(torch.full_like(yaw, float(target_yaw)) - yaw)
    desired_cmd[:, 2] = torch.clamp(args_cli.yaw_kp * yaw_error, -args_cli.max_wz, args_cli.max_wz)
    desired_cmd[:, 3] = float(term.cfg.default_body_height)
    desired_cmd[:, 4] = 0.0

    close = xy_dist < float(args_cli.stop_radius)
    desired_cmd[close, :3] = 0.0

    action = torch.zeros(env.action_space.shape, device=device)
    action[:, :5] = _normalize_base_action(term, desired_cmd)
    action[:, 5:11] = _normalize_arm_action(term, arm_q_target)
    return action, desired_cmd, xy_dist, yaw_error


def _draw_debug(draw, env, base_target_xy: torch.Tensor, ee_target_w: torch.Tensor):
    if draw is None:
        return
    robot = env.unwrapped.scene["robot"]
    root = robot.data.root_pos_w[0]
    ee_pos = _get_body_pos_w(robot, "gripper_grasp_center")
    ee = ee_pos[0] if ee_pos is not None else None

    base_target = (float(base_target_xy[0]), float(base_target_xy[1]), 0.04)
    ee_target = tuple(float(x) for x in ee_target_w.tolist())
    base_now = (float(root[0]), float(root[1]), 0.08)

    draw.clear_points()
    draw.clear_lines()
    points = [base_target, ee_target]
    colors = [(1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0)]
    sizes = [18.0, 14.0]
    if ee is not None:
        points.append(tuple(float(x) for x in ee.tolist()))
        colors.append((0.0, 1.0, 1.0, 1.0))
        sizes.append(12.0)
    draw.draw_points(points, colors, sizes)

    starts = [base_now]
    ends = [base_target]
    line_colors = [(1.0, 0.85, 0.0, 1.0)]
    widths = [2.0]
    if ee is not None:
        starts.append(tuple(float(x) for x in ee.tolist()))
        ends.append(ee_target)
        line_colors.append((0.0, 1.0, 1.0, 1.0))
        widths.append(2.0)
    draw.draw_lines(starts, ends, line_colors, widths)


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)
    term = _get_action_term(env)
    draw = _make_draw()

    print(f"[INFO] observation space: {env.observation_space}")
    print(f"[INFO] action space: {env.action_space}")
    print(f"[INFO] action term dim: {term.action_dim}; first 5 = base, last 6 = arm")
    print(f"[INFO] base target xy: {tuple(args_cli.base_target)}; ee debug target: {tuple(args_cli.ee_target)}")

    env.reset()
    device = env.unwrapped.device
    base_target_xy = torch.tensor(args_cli.base_target, device=device, dtype=torch.float32)
    ee_target_w = torch.tensor(args_cli.ee_target, device=device, dtype=torch.float32)
    arm_q_target = torch.tensor(args_cli.arm_q_target, device=device, dtype=torch.float32).view(1, 6)
    arm_q_target = arm_q_target.repeat(env.unwrapped.num_envs, 1)

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            action, cmd, xy_dist, yaw_error = _build_debug_action(
                env, term, base_target_xy, float(args_cli.base_target_yaw), arm_q_target
            )
            obs, rew, terminated, truncated, info = env.step(action)
            _draw_debug(draw, env, base_target_xy, ee_target_w)

            if step % max(args_cli.print_every, 1) == 0:
                robot = env.unwrapped.scene["robot"]
                ee_pos = _get_body_pos_w(robot, "gripper_grasp_center")
                ee_msg = "unavailable"
                if ee_pos is not None:
                    ee_dist = torch.linalg.norm(ee_pos[0] - ee_target_w).item()
                    ee_msg = f"pos={ee_pos[0].detach().cpu().tolist()} dist_to_debug_target={ee_dist:.3f}"
                print(
                    f"[DBG] step={step} base_xy={robot.data.root_pos_w[0, :2].detach().cpu().tolist()} "
                    f"dist={xy_dist[0].item():.3f} yaw_err={yaw_error[0].item():.3f} "
                    f"cmd={cmd[0, :3].detach().cpu().tolist()} ee={ee_msg}"
                )

            step += 1
            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
