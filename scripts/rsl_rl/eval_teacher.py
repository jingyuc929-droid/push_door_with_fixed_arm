# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic evaluation for a DoorBot RSL-RL teacher checkpoint."""

import argparse
import csv
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


def _str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


parser = argparse.ArgumentParser(description="Deterministically evaluate a DoorBot PPO teacher checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record evaluation video.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate, e.g. 64.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--agent", type=str, default="rsl_rl_teacher_cfg_entry_point", help="RL agent config entry point.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--eval_episodes", type=int, default=200, help="Number of complete episodes to evaluate.")
parser.add_argument(
    "--disable_staged_reset",
    type=_str_to_bool,
    nargs="?",
    const=True,
    default=True,
    help="Disable staged reset starts.",
)
parser.add_argument(
    "--deterministic",
    type=_str_to_bool,
    nargs="?",
    const=True,
    default=True,
    help="Use deterministic inference policy.",
)
parser.add_argument("--door_joint_name", type=str, default="door_joint", help="Door joint used for success detection.")
parser.add_argument("--door_closed_pos", type=float, default=0.0, help="Door closed joint position.")
parser.add_argument("--door_open_sign", type=float, default=1.0, help="Door open sign.")
parser.add_argument("--door_open_threshold", type=float, default=0.30, help="Door open success threshold.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.task is None:
    parser.error("--task is required, for example: --task Template-Door-Env-v0")

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import door_env.tasks  # noqa: F401


def _disable_staged_reset_in_cfg(env_cfg) -> bool:
    changed = False
    candidates = []
    events = getattr(env_cfg, "events", None)
    if events is not None:
        for name in ("staged_reset", "reset_staged", "stage_reset"):
            if hasattr(events, name):
                candidates.append(getattr(events, name))
    for event_term in candidates:
        params = getattr(event_term, "params", None)
        if isinstance(params, dict):
            for key in ("p_grasp_start", "p_unlock_start", "p_opening_start"):
                if key in params:
                    params[key] = 0.0
                    changed = True
    return changed


def _find_tensor_by_key(obj, key: str):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) == key and torch.is_tensor(v):
                return v
            found = _find_tensor_by_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found = _find_tensor_by_key(item, key)
            if found is not None:
                return found
    return None


def _termination_mask(base_env, extras, name: str, dones: torch.Tensor) -> torch.Tensor:
    for key in (
        name,
        f"Episode_Termination/{name}",
        f"termination/{name}",
        f"terminated/{name}",
        f"truncated/{name}",
    ):
        found = _find_tensor_by_key(extras, key)
        if found is not None and found.numel() == dones.numel():
            return found.to(device=dones.device, dtype=torch.bool).reshape_as(dones)

    term_manager = getattr(base_env, "termination_manager", None)
    if term_manager is not None:
        for attr_name in ("terminated", "truncated", "_term_dones", "_trunc_dones"):
            container = getattr(term_manager, attr_name, None)
            if isinstance(container, dict) and name in container and torch.is_tensor(container[name]):
                value = container[name]
                if value.numel() == dones.numel():
                    return value.to(device=dones.device, dtype=torch.bool).reshape_as(dones)
        for method_name in ("get_term", "get_active_iterable_terms"):
            method = getattr(term_manager, method_name, None)
            if method is not None:
                try:
                    value = method(name)
                    if torch.is_tensor(value) and value.numel() == dones.numel():
                        return value.to(device=dones.device, dtype=torch.bool).reshape_as(dones)
                except Exception:
                    pass

    return torch.zeros_like(dones, dtype=torch.bool)


def _resolve_door_joint_id(door, joint_name: str) -> int:
    joint_names = list(door.data.joint_names)
    if joint_name in joint_names:
        return joint_names.index(joint_name)
    candidates = [i for i, name in enumerate(joint_names) if joint_name.lower() in name.lower()]
    if candidates:
        return candidates[0]
    raise RuntimeError(f"Could not find door joint {joint_name!r}. Available joints: {joint_names}")


def _door_open(door, joint_id: int, closed_pos: float, open_sign: float) -> torch.Tensor:
    pos = door.data.joint_pos[:, joint_id]
    return float(open_sign) * (pos - float(closed_pos))


def _write_results_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "episode_id",
        "env_id",
        "success",
        "episode_length",
        "final_door_open",
        "max_door_open",
        "timeout",
        "release_after_grasp_failure",
        "release_after_unlock_failure",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    staged_reset_disabled = False
    if args_cli.disable_staged_reset:
        staged_reset_disabled = _disable_staged_reset_in_cfg(env_cfg)
        if not staged_reset_disabled:
            print("[WARN] Requested --disable_staged_reset=True, but staged reset probability fields were not found.")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    checkpoint_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = checkpoint_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(checkpoint_dir, "videos", "eval_teacher"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base_env = env.unwrapped
    door = base_env.scene["door"]
    door_joint_id = _resolve_door_joint_id(door, args_cli.door_joint_name)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base_env.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    obs = env.get_observations()
    num_envs = env.num_envs
    episode_lengths = torch.zeros(num_envs, dtype=torch.int32, device=base_env.device)
    max_door_open = _door_open(door, door_joint_id, args_cli.door_closed_pos, args_cli.door_open_sign).clone()

    rows = []
    total_episodes = 0
    success_episodes = 0
    timeout_episodes = 0
    release_after_grasp_episodes = 0
    release_after_unlock_episodes = 0
    next_progress_episodes = 20

    with torch.inference_mode():
        while simulation_app.is_running() and total_episodes < int(args_cli.eval_episodes):
            actions = policy(obs)
            door_open_before = _door_open(door, door_joint_id, args_cli.door_closed_pos, args_cli.door_open_sign)
            max_door_open = torch.maximum(max_door_open, door_open_before)

            obs, _rew, dones, extras = env.step(actions)

            door_open_after = _door_open(door, door_joint_id, args_cli.door_closed_pos, args_cli.door_open_sign)
            max_door_open = torch.maximum(max_door_open, door_open_after)
            episode_lengths += 1

            timeout_mask = _termination_mask(base_env, extras, "time_out", dones)
            release_grasp_mask = _termination_mask(base_env, extras, "release_after_grasp_failure", dones)
            release_unlock_mask = _termination_mask(base_env, extras, "release_after_unlock_failure", dones)

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            for env_id_t in done_ids:
                if total_episodes >= int(args_cli.eval_episodes):
                    break
                env_id = int(env_id_t.item())
                final_open = float(torch.maximum(door_open_before[env_id], door_open_after[env_id]).item())
                max_open = float(max_door_open[env_id].item())
                success = max(final_open, max_open) >= float(args_cli.door_open_threshold)
                timeout = bool(timeout_mask[env_id].item())
                release_grasp = bool(release_grasp_mask[env_id].item())
                release_unlock = bool(release_unlock_mask[env_id].item())

                rows.append(
                    {
                        "episode_id": total_episodes,
                        "env_id": env_id,
                        "success": int(success),
                        "episode_length": int(episode_lengths[env_id].item()),
                        "final_door_open": final_open,
                        "max_door_open": max_open,
                        "timeout": int(timeout),
                        "release_after_grasp_failure": int(release_grasp),
                        "release_after_unlock_failure": int(release_unlock),
                    }
                )

                total_episodes += 1
                success_episodes += int(success)
                timeout_episodes += int(timeout)
                release_after_grasp_episodes += int(release_grasp)
                release_after_unlock_episodes += int(release_unlock)

                episode_lengths[env_id] = 0
                max_door_open[env_id] = door_open_after[env_id]

            if done_ids.numel() > 0:
                policy_nn.reset(dones)

            while total_episodes >= next_progress_episodes:
                print(
                    f"Evaluated {total_episodes}/{args_cli.eval_episodes} episodes, "
                    f"success_rate={100.0 * success_episodes / max(1, total_episodes):.2f}%"
                )
                next_progress_episodes += 20

            if args_cli.video and len(rows) > 0:
                break

    eval_dir = os.path.join("logs", "eval", datetime.now().strftime("%Y-%m-%d_%H-%M-%S_teacher"))
    csv_path = os.path.abspath(os.path.join(eval_dir, "eval_teacher_results.csv"))
    _write_results_csv(csv_path, rows)

    total = max(1, len(rows))
    other_failures = sum(
        1
        for row in rows
        if not row["success"]
        and not row["timeout"]
        and not row["release_after_grasp_failure"]
        and not row["release_after_unlock_failure"]
    )
    mean_episode_length = sum(row["episode_length"] for row in rows) / total
    mean_final_open = sum(row["final_door_open"] for row in rows) / total
    mean_max_open = sum(row["max_door_open"] for row in rows) / total
    max_open_all = max((row["max_door_open"] for row in rows), default=0.0)

    print("\n========== Deterministic Teacher Evaluation ==========")
    print(f"Checkpoint: {resume_path}")
    print(f"Num envs: {num_envs}")
    print(f"Eval episodes: {len(rows)}")
    print(f"Staged reset disabled: {staged_reset_disabled}")
    print(f"Deterministic policy: {args_cli.deterministic}")
    print("")
    print(f"Success episodes: {success_episodes}")
    print(f"Success rate: {100.0 * success_episodes / total:.2f} %")
    print("")
    print(f"Timeout episodes: {timeout_episodes}")
    print(f"Release after grasp failure: {release_after_grasp_episodes}")
    print(f"Release after unlock failure: {release_after_unlock_episodes}")
    print(f"Other failure: {other_failures}")
    print("")
    print(f"Mean episode length: {mean_episode_length:.2f}")
    print(f"Mean final door_open: {mean_final_open:.4f}")
    print(f"Mean max door_open: {mean_max_open:.4f}")
    print(f"Max door_open over all episodes: {max_open_all:.4f}")
    print(f"CSV: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
