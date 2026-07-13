# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import door_env.tasks  # noqa: F401


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
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
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return _quat_mul(_quat_mul(q, qv), _quat_conjugate(q))[..., 1:]


def _compute_doorway_center_debug(
    base_env,
    doorway_center_xy: tuple[float, float],
    doorway_z: float = 0.08,
    step_count: int = 0,
):
    robot = base_env.scene["robot"]
    door = base_env.scene["door"]
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    door_pos_w = door.data.root_pos_w
    door_quat_w = door.data.root_quat_w

    center_d = torch.tensor(
        (float(doorway_center_xy[0]), float(doorway_center_xy[1]), float(doorway_z)),
        device=base_pos_w.device,
        dtype=base_pos_w.dtype,
    ).view(1, 3).repeat(base_pos_w.shape[0], 1)
    center_w = door_pos_w + _quat_rotate(door_quat_w, center_d)
    center_b = _quat_rotate(_quat_conjugate(base_quat_w), center_w - base_pos_w)
    center_reproject_w = base_pos_w + _quat_rotate(base_quat_w, center_b)
    reproj_err = torch.linalg.norm(center_reproject_w - center_w, dim=-1)

    if step_count % 120 == 0:
        print(f"[DEBUG]: doorway center base->world reprojection max error: {float(reproj_err.max().item()):.6e}")
    return center_b, reproj_err


def main():
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    center_labels = []
    if not args_cli.headless:
        try:
            import omni.ui as ui

            hud_win = ui.Window("Zero Agent Doorway Center HUD", width=560, height=260)
            with hud_win.frame:
                with ui.VStack(spacing=4):
                    ui.Label("doorway center in robot base frame; err is base->world reprojection")
                    for env_id in range(env.unwrapped.num_envs):
                        center_labels.append(ui.Label(f"env{env_id}: center_b (--, --, --), err --"))
            print("[INFO]: Doorway center HUD enabled.")
        except Exception as exc:
            print(f"[WARN]: Doorway center HUD disabled: {exc}")

    traverse_success_params = getattr(getattr(env_cfg.terminations, "base_traverse_success", None), "params", {})
    doorway_center_xy = traverse_success_params.get("doorway_center_xy", (0.6, 0.0))

    step_count = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # compute zero actions
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            env.step(actions)
            if center_labels and step_count % 5 == 0:
                center_b, reproj_err = _compute_doorway_center_debug(
                    env.unwrapped, doorway_center_xy, step_count=step_count
                )
                for env_id, label in enumerate(center_labels):
                    if env_id < center_b.shape[0]:
                        label.text = (
                            f"env{env_id}: center_b "
                            f"({float(center_b[env_id, 0].item()):+.3f}, "
                            f"{float(center_b[env_id, 1].item()):+.3f}, "
                            f"{float(center_b[env_id, 2].item()):+.3f}), "
                            f"err {float(reproj_err[env_id].item()):.1e}"
                        )
            step_count += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
