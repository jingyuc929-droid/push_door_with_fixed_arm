"""Observation terms for hierarchical pick-and-place tasks."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_inv, quat_mul


def _zeros(env, dim: int) -> torch.Tensor:
    return torch.zeros((env.num_envs, dim), device=env.device, dtype=torch.float32)


def _get_asset(env, asset_cfg: SceneEntityCfg):
    try:
        return env.scene[asset_cfg.name]
    except Exception:
        return None


def high_level_base_command(env) -> torch.Tensor:
    return getattr(env, "high_level_base_command", _zeros(env, 5))


def high_level_previous_action(env) -> torch.Tensor:
    action = getattr(env, "high_level_previous_action", None)
    if action is None:
        return _zeros(env, getattr(env.cfg.config_summary.env, "num_actions", 0))
    return action


def low_level_last_action(env, dim: int = 12) -> torch.Tensor:
    action = getattr(env, "low_level_last_action", None)
    if action is None:
        return _zeros(env, dim)
    return action


def base_pitch_roll_height(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w)
    height = asset.data.root_pos_w[:, 2]
    return torch.stack((roll, pitch, height), dim=-1)


def ee_pose_in_base(
    env,
    ee_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ee_asset: Articulation = env.scene[ee_asset_cfg.name]
    robot: Articulation = env.scene[robot_asset_cfg.name]
    if len(ee_asset_cfg.body_ids) < 1:
        raise ValueError("ee_asset_cfg must resolve at least one body.")
    body_ids = ee_asset_cfg.body_ids
    ee_pos_w = ee_asset.data.body_pos_w[:, body_ids].mean(dim=1)
    ee_quat_w = ee_asset.data.body_quat_w[:, body_ids[0]]
    ee_pos_b = quat_apply_inverse(robot.data.root_quat_w, ee_pos_w - robot.data.root_pos_w)
    ee_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), ee_quat_w)
    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)


def ee_position_in_base(
    env,
    ee_asset_cfg: SceneEntityCfg,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return ee_pose_in_base(env, ee_asset_cfg, robot_asset_cfg)[:, :3]


def object_position_in_base(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject | None = _get_asset(env, object_cfg)
    if obj is None:
        return _zeros(env, 3)
    robot: Articulation = env.scene[robot_asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def object_velocity_in_base(
    env,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    obj: RigidObject | None = _get_asset(env, object_cfg)
    if obj is None:
        return _zeros(env, 3)
    robot: Articulation = env.scene[robot_asset_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)


def pick_position_in_base(env) -> torch.Tensor:
    pick_pos_w = getattr(env, "pick_place_pick_pos_w", None)
    if pick_pos_w is None:
        return _zeros(env, 3)
    robot: Articulation = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, pick_pos_w - robot.data.root_pos_w)


def target_position_in_base(env) -> torch.Tensor:
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        return pick_position_in_base(env)
    pick_b = pick_position_in_base(env)
    place_b = place_position_in_base(env)
    use_place = phase.to(device=env.device).view(-1, 1) >= 1
    return torch.where(use_place, place_b, pick_b)


def place_position_in_base(env) -> torch.Tensor:
    place_pos_w = getattr(env, "pick_place_place_pos_w", None)
    if place_pos_w is None:
        return _zeros(env, 3)
    robot: Articulation = env.scene["robot"]
    return quat_apply_inverse(robot.data.root_quat_w, place_pos_w - robot.data.root_pos_w)


def phase_index(env, num_phases: int = 6, one_hot: bool = True) -> torch.Tensor:
    phase = getattr(env, "pick_place_phase", None)
    if phase is None:
        phase = torch.zeros((env.num_envs,), device=env.device, dtype=torch.long)
    phase = phase.to(device=env.device, dtype=torch.long).clamp(0, int(num_phases) - 1)
    if not one_hot:
        return phase.to(dtype=torch.float32).unsqueeze(-1) / max(float(num_phases - 1), 1.0)
    return torch.nn.functional.one_hot(phase, num_classes=int(num_phases)).to(dtype=torch.float32)


def object_in_gripper_flag(
    env,
    ee_asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    threshold: float = 0.08,
) -> torch.Tensor:
    obj_b = object_position_in_base(env, object_cfg)
    ee_b = ee_position_in_base(env, ee_asset_cfg)
    return (torch.linalg.norm(obj_b - ee_b, dim=-1, keepdim=True) < float(threshold)).to(dtype=torch.float32)
