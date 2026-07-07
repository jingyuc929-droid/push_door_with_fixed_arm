# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import argparse


# 只允许在 SimulationApp 启动前导入这个
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Smoke test for ARX5 MIT-style arm action.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument(
    "--mode",
    type=str,
    default="zero",
    choices=[
        "zero",
        "joint1_pos",
        "joint1_neg",
        "joint2_pos",
        "joint2_neg",
        "joint3_pos",
        "joint3_neg",
        "joint4_pos",
        "joint4_neg",
        "joint5_pos",
        "joint5_neg",
        "joint6_pos",
        "joint6_neg",
    ],
    help="Smoke-test action mode.",
)
parser.add_argument("--steps", type=int, default=3000, help="Total env steps.")
parser.add_argument("--gripper", type=float, default=0.0, help="Binary gripper action placeholder.")
parser.add_argument("--amp", type=float, default=0.2, help="Action amplitude for tested joint.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 先启动 SimulationApp
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ===== 下面这些导入必须放在 SimulationApp 之后 =====
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import door_env.tasks  # noqa: F401


def make_action(action_dim: int, arm_cmd, gripper_cmd=0.0, device="cuda"):
    """Build a single-env fixed action tensor."""
    a = torch.zeros((1, action_dim), device=device, dtype=torch.float32)
    n_arm = min(6, action_dim)
    a[0, :n_arm] = torch.tensor(arm_cmd[:n_arm], dtype=torch.float32, device=device)
    if action_dim > 6:
        a[0, 6] = float(gripper_cmd)
    return a


def get_action_dim(env) -> int:
    """Try a few ways to get total action dimension."""
    # 1) manager-based env
    try:
        return int(env.unwrapped.action_manager.total_action_dim)
    except Exception:
        pass

    # 2) gym action space
    try:
        shape = env.action_space.shape
        if shape is not None and len(shape) > 0:
            return int(shape[-1])
    except Exception:
        pass

    raise RuntimeError("Cannot determine action dimension from env/action_manager.")


def build_fixed_arm_cmd(mode: str, amp: float):
    arm = [0.0] * 6
    mapping = {
        "joint1_pos": (0, +amp),
        "joint1_neg": (0, -amp),
        "joint2_pos": (1, +amp),
        "joint2_neg": (1, -amp),
        "joint3_pos": (2, +amp),
        "joint3_neg": (2, -amp),
        "joint4_pos": (3, +amp),
        "joint4_neg": (3, -amp),
        "joint5_pos": (4, +amp),
        "joint5_neg": (4, -amp),
        "joint6_pos": (5, +amp),
        "joint6_neg": (5, -amp),
    }
    if mode in mapping:
        jid, val = mapping[mode]
        arm[jid] = val
    return arm


def safe_get_mit_term(base_env):
    """Try to fetch the custom MIT action term."""
    try:
        return base_env.action_manager.get_term("arm_action")
    except Exception:
        pass
    try:
        return base_env.action_manager._terms["arm_action"]  # fallback for some versions
    except Exception:
        pass
    return None


def main():
    # parse env cfg
    device = getattr(args_cli, "device", "cuda:0")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device= device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
    )
    env_cfg.scene.num_envs = 1

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    device = base_env.device

    obs, _ = env.reset()

    action_dim = get_action_dim(env)
    print(f"[INFO] action_dim = {action_dim}")

    robot = base_env.scene["robot"]
    mit_term = safe_get_mit_term(base_env)

    arm_cmd = build_fixed_arm_cmd(args_cli.mode, args_cli.amp)
    action = make_action(action_dim, arm_cmd, gripper_cmd=args_cli.gripper, device=device)

    print("[INFO] smoke test configuration:")
    print(f"       mode     = {args_cli.mode}")
    print(f"       amp      = {args_cli.amp}")
    print(f"       steps    = {args_cli.steps}")
    print(f"       arm_cmd  = {arm_cmd}")
    print(f"       gripper  = {args_cli.gripper}")
    print(f"       device   = {device}")
    print(f"       mit_term = {'found' if mit_term is not None else 'NOT found'}")

    for step in range(args_cli.steps):
        obs, rew, terminated, truncated, info = env.step(action)

        if step % 100 == 0:
            q = robot.data.joint_pos[0, :6].detach().cpu()
            dq = robot.data.joint_vel[0, :6].detach().cpu()

            print(f"\n[step {step}]")
            print("q    =", [round(x.item(), 5) for x in q])
            print("dq   =", [round(x.item(), 5) for x in dq])

            if mit_term is not None:
                try:
                    q_des = mit_term.q_des[0].detach().cpu()
                    tau = mit_term.tau_cmd[0].detach().cpu()
                    q_err = (mit_term.q_des[0] - robot.data.joint_pos[0, :6]).detach().cpu()

                    print("q_des=", [round(x.item(), 5) for x in q_des])
                    print("q_err=", [round(x.item(), 5) for x in q_err])
                    print("tau  =", [round(x.item(), 5) for x in tau])
                except Exception as e:
                    print(f"[WARN] MIT debug read failed: {e}")

        done = bool(terminated[0]) or bool(truncated[0])
        if done:
            print(f"[INFO] env reset at step {step}")
            obs, _ = env.reset()

    print("[INFO] smoke test finished.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()