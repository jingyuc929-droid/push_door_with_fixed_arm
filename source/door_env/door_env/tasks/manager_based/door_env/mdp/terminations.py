from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation

from .rewards import (
    _physical_door_unlocked,
    fingertip_mid_to_handle_grasp_point_distance,
    align_grasp_pose_v2,
    filtered_contact_force_norm,
    filtered_contact_force_vec_w,
    )


# -------------------------
# Quaternion helpers (wxyz)
# -------------------------
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


# -------------------------
# Pose helpers
# -------------------------
def _body_pose_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[cfg.name]
    bid = cfg.body_ids[0]
    pos = asset.data.body_pos_w[:, bid, :]
    quat = asset.data.body_quat_w[:, bid, :]  # wxyz
    return pos, quat


def fingertip_mid_to_handle_distance(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pTip = 0.5 * (pL + pR)
    pH, _ = _body_pose_w(env, handle_cfg)
    return torch.linalg.norm(pTip - pH, dim=-1)


def align_grasp_around_handle_local(
    env: ManagerBasedRLEnv,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    grasp_axis: int = 2,
    min_sep: float = 0.002,
) -> torch.Tensor:
    """Return 1 when left/right fingertips are on opposite sides of handle along handle-frame grasp_axis."""
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pH, qH = _body_pose_w(env, handle_cfg)

    L_h = quat_rotate(quat_conjugate(qH), pL - pH)
    R_h = quat_rotate(quat_conjugate(qH), pR - pH)

    side = (L_h[:, grasp_axis] * R_h[:, grasp_axis]) < 0.0
    sep = torch.abs(L_h[:, grasp_axis] - R_h[:, grasp_axis]) > min_sep
    return (side & sep).float()


# -------------------------
# Filtered (handle-only) contact force history
# -------------------------
def _filtered_contact_force_mag_history(sensor: ContactSensor) -> torch.Tensor:
    """Return *filtered* contact force magnitude history per env: [N, H].

    IMPORTANT:
      - When ContactSensorCfg(filter_prim_paths_expr=[...]) is used, the per-filtered-prim forces are in
        `force_matrix_w_history`.
      - `net_forces_w_history` remains the TOTAL contact force history and includes self-collision / door panel, etc.

    This helper is intentionally **safe-fail**:
      - If `force_matrix_w_history` is not available, it returns zeros rather than falling back to total contacts.
        This prevents accidental success triggers from self-collision noise.
    """
    fm = getattr(sensor.data, "force_matrix_w_history", None)
    if fm is None:
        return torch.zeros((sensor.data.net_forces_w_history.shape[0], sensor.data.net_forces_w_history.shape[1]),
                           device=sensor.data.net_forces_w_history.device)

    # Common shapes:
    #  - [N, H, B, K, 3]  (B bodies tracked by sensor; K filtered prims)
    #  - [N, H, K, 3]
    if fm.ndim == 5:
        vec = fm.sum(dim=3)                          # [N, H, B, 3]
        mag = torch.linalg.norm(vec, dim=-1)         # [N, H, B]
        return mag.max(dim=2).values                 # [N, H]
    if fm.ndim == 4:
        vec = fm.sum(dim=2)                          # [N, H, 3]
        return torch.linalg.norm(vec, dim=-1)        # [N, H]

    raise RuntimeError(f"Unexpected force_matrix_w_history shape: {tuple(fm.shape)}")


def sustained_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_steps: int = 3,
    threshold: float = 2.0,
    require_full_history: bool = True,
) -> torch.Tensor:
    """True if filtered (handle-only) contact force > threshold for the last `num_steps` samples."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    mag_hist = _filtered_contact_force_mag_history(sensor)  # [N, H]

    H = mag_hist.shape[1]
    if require_full_history and H < num_steps:
        return torch.zeros((mag_hist.shape[0],), dtype=torch.bool, device=mag_hist.device)

    k = num_steps if require_full_history else min(num_steps, H)
    tail = mag_hist[:, -k:]  # [N, k]
    return torch.all(tail > threshold, dim=-1)


def sustained_two_sensors_contact(
    env: ManagerBasedRLEnv,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    num_steps: int = 3,
    threshold: float = 2.0,
    require_full_history: bool = True,
) -> torch.Tensor:
    """True if BOTH sensors sustain filtered contact for `num_steps`."""
    left_ok = sustained_contact(env, left_sensor_cfg, num_steps=num_steps, threshold=threshold,
                               require_full_history=require_full_history)
    right_ok = sustained_contact(env, right_sensor_cfg, num_steps=num_steps, threshold=threshold,
                                require_full_history=require_full_history)
    return left_ok & right_ok


# -------------------------
# Task-specific success termination: grasp handle
# -------------------------
def grasp_handle_sustained(
    env: ManagerBasedRLEnv,
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    # history / contact
    num_steps: int = 5,
    force_threshold: float = 1.0,
    require_any_finger_contact: bool = False,
    use_force_norm: bool = True,
    # near gate
    distance_threshold: float = 0.10,
    use_grasp_point: bool = True,
    handle_offset_h: tuple[float, float, float] =  (-0.09, 0.04, 0.01),
    # wrap gate
    require_wrap: bool = True,
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    align_side_weight: float = 0.70,
    align_open_weight: float = 0.10,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,
    # closure
    open_width: float = 0.08,
    min_closedness: float = 0.5,
) -> torch.Tensor:
    """Grasp success termination aligned with current ARX5 rewards/grasp_success gate."""

    # ------------------------------------------------------------------
    # 1) sustained filtered contact history
    # ------------------------------------------------------------------
    left_hist_ok = sustained_contact(
        env,
        left_sensor_cfg,
        num_steps=num_steps,
        threshold=force_threshold,
        require_full_history=True,
    )
    right_hist_ok = sustained_contact(
        env,
        right_sensor_cfg,
        num_steps=num_steps,
        threshold=force_threshold,
        require_full_history=True,
    )

    if require_any_finger_contact:
        contact_hist_ok = left_hist_ok | right_hist_ok
    else:
        contact_hist_ok = left_hist_ok & right_hist_ok

    # ------------------------------------------------------------------
    # 2) near gate
    # ------------------------------------------------------------------
    if use_grasp_point:
        dist = fingertip_mid_to_handle_grasp_point_distance(
            env,
            left_finger_cfg,
            right_finger_cfg,
            handle_cfg,
            handle_offset_h=handle_offset_h,
        )
    else:
        dist = fingertip_mid_to_handle_distance(
            env,
            left_finger_cfg,
            right_finger_cfg,
            handle_cfg,
        )
    near_ok = dist < distance_threshold

    # ------------------------------------------------------------------
    # 3) wrap/alignment gate
    # ------------------------------------------------------------------
    if require_wrap:
        wrap = align_grasp_pose_v2(
            env=env,
            left_finger_cfg=left_finger_cfg,
            right_finger_cfg=right_finger_cfg,
            handle_cfg=handle_cfg,
            hand_cfg=hand_cfg,
            grasp_axis=grasp_axis,
            min_sep=min_sep,
            sep_scale=sep_scale,
            symmetry_scale=symmetry_scale,
            gripper_open_axis_hand=gripper_open_axis_hand,
            gripper_approach_axis_hand=gripper_approach_axis_hand,
            handle_approach_axis=handle_approach_axis,
            side_weight=align_side_weight,
            open_weight=align_open_weight,
            approach_weight=align_approach_weight,
        )
        wrap_ok = wrap >= align_threshold
    else:
        wrap_ok = torch.ones_like(near_ok, dtype=torch.bool)

    # ------------------------------------------------------------------
    # 4) instantaneous force gate (same style as grasp_success reward)
    # ------------------------------------------------------------------
    sL = env.scene[left_sensor_cfg.name]
    sR = env.scene[right_sensor_cfg.name]

    if use_force_norm:
        fL = filtered_contact_force_norm(sL)
        fR = filtered_contact_force_norm(sR)
    else:
        _, qH = _body_pose_w(env, handle_cfg)
        qHc = quat_conjugate(qH)
        FL_h = quat_rotate(qHc, filtered_contact_force_vec_w(sL))
        FR_h = quat_rotate(qHc, filtered_contact_force_vec_w(sR))
        fL = torch.abs(FL_h[:, grasp_axis])
        fR = torch.abs(FR_h[:, grasp_axis])

    if require_any_finger_contact:
        force_ok = torch.maximum(fL, fR) > force_threshold
    else:
        force_ok = torch.minimum(fL, fR) > force_threshold

    # ------------------------------------------------------------------
    # 5) closure gate
    # ------------------------------------------------------------------
    robot: Articulation = env.scene[gripper_cfg.name]
    width = robot.data.joint_pos[:, gripper_cfg.joint_ids].sum(dim=-1)
    closedness = 1.0 - torch.clamp(width / open_width, 0.0, 1.0)
    close_ok = closedness > min_closedness

    # ------------------------------------------------------------------
    # final gate
    # ------------------------------------------------------------------
    return contact_hist_ok & near_ok & wrap_ok & force_ok & close_ok


def unlock_handle_after_grasp(
    env: ManagerBasedRLEnv,
    # --- gate: contact + closure (+ optional near) ---
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    num_steps: int = 2,
    force_threshold: float = 1.0,
    require_any_contact: bool = True,     # True: 单指即可；False: 必须双指
    require_near: bool = True,
    distance_threshold: float = 0.12,     # 解锁阶段放宽（你 grasp 用的是 0.08）
    open_width: float = 0.08,
    min_closedness: float = 0.4,         

    # --- unlock check: handle joint threshold ---
    handle_joint_cfg: SceneEntityCfg = None,
    handle_threshold: float = -0.2,
    less_than: bool = True,
) -> torch.Tensor:
    """Success termination for 'unlock handle' with relaxed gate.

    Gate (recommended for unlock stage):
      - handle-only filtered contact sustained on ANY finger (or BOTH if require_any_contact=False)
      - gripper closedness above min_closedness
      - optional: fingertip-mid near handle (with looser threshold)

    Success:
      - handle joint position crosses handle_threshold.
    """
    # 1) contact gate (handle-only filtered contact history)
    left_ok = sustained_contact(env, left_sensor_cfg, num_steps=num_steps, threshold=force_threshold, require_full_history=True)
    right_ok = sustained_contact(env, right_sensor_cfg, num_steps=num_steps, threshold=force_threshold, require_full_history=True)
    contact_ok = (left_ok | right_ok) if require_any_contact else (left_ok & right_ok)

    # 2) closedness gate (looser than grasp)
    robot: Articulation = env.scene[gripper_cfg.name]
    width = robot.data.joint_pos[:, gripper_cfg.joint_ids].sum(dim=-1)  # open ~ open_width
    closedness = 1.0 - torch.clamp(width / open_width, 0.0, 1.0)
    close_ok = closedness > min_closedness

    # 3) optional near gate (looser than grasp)
    if require_near:
        dist = fingertip_mid_to_handle_distance(env, left_finger_cfg, right_finger_cfg, handle_cfg)
        near_ok = dist < distance_threshold
        gate_ok = contact_ok & close_ok & near_ok
    else:
        gate_ok = contact_ok & close_ok

    # 4) unlock check: handle joint threshold
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    unlocked = (handle_pos < handle_threshold) if less_than else (handle_pos > handle_threshold)
    return gate_ok & unlocked




def door_open_success_only(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_threshold: float = 0.35,
    num_steps: int = 3,
) -> torch.Tensor:
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    good = door_open >= float(door_open_threshold)

    N = door_open.shape[0]
    device = door_open.device

    if (not hasattr(env, "_door_open_success_counter")) or (env._door_open_success_counter.shape[0] != N):
        env._door_open_success_counter = torch.zeros(N, device=device, dtype=torch.int32)

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_door_open_success")) or (env._prev_ep_len_door_open_success.shape[0] != N):
            env._prev_ep_len_door_open_success = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_door_open_success) | (env.episode_length_buf == 0)
        env._prev_ep_len_door_open_success = env.episode_length_buf.clone()

    if reset_mask is not None:
        env._door_open_success_counter = torch.where(
            reset_mask,
            torch.zeros_like(env._door_open_success_counter),
            env._door_open_success_counter,
        )

    env._door_open_success_counter = torch.where(
        good,
        env._door_open_success_counter + 1,
        torch.zeros_like(env._door_open_success_counter),
    )

    return env._door_open_success_counter >= int(max(1, num_steps))


def release_after_grasp_failure(
    env: ManagerBasedRLEnv,
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.25,
    require_any_contact: bool = False,
    open_width: float = 0.09,
    min_closedness: float = 0.35,
    distance_threshold: float = 0.13,
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, 0.01),
    grace_steps: int = 10,
    fail_steps: int = 25,
) -> torch.Tensor:
    N = env.num_envs
    device = env.device

    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(N, dtype=torch.bool, device=device)
    physical_unlocked = _physical_door_unlocked(env)

    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (
        (torch.maximum(fL, fR) > float(contact_threshold))
        if require_any_contact
        else (torch.minimum(fL, fR) > float(contact_threshold))
    )

    robot: Articulation = env.scene[gripper_cfg.name]
    width = robot.data.joint_pos[:, gripper_cfg.joint_ids].sum(dim=-1)
    closedness = 1.0 - torch.clamp(width / float(open_width), 0.0, 1.0)
    close_ok = closedness > float(min_closedness)

    dist = fingertip_mid_to_handle_grasp_point_distance(
        env,
        left_finger_cfg,
        right_finger_cfg,
        handle_cfg,
        handle_offset_h=handle_offset_h,
    )
    near_ok = dist < float(distance_threshold)
    keep_ok = contact_ok & close_ok & near_ok

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_release_after_grasp")) or (
            env._prev_ep_len_release_after_grasp.shape[0] != N
        ):
            env._prev_ep_len_release_after_grasp = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_release_after_grasp)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_release_after_grasp = env.episode_length_buf.clone()

    if (not hasattr(env, "_release_after_grasp_counter")) or (
        env._release_after_grasp_counter.shape[0] != N
    ):
        env._release_after_grasp_counter = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_release_after_grasp_grace")) or (
        env._release_after_grasp_grace.shape[0] != N
    ):
        env._release_after_grasp_grace = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_prev_release_after_grasp_grasp_ok")) or (
        env._prev_release_after_grasp_grasp_ok.shape[0] != N
    ):
        env._prev_release_after_grasp_grasp_ok = torch.zeros(N, dtype=torch.bool, device=device)

    newly_grasped = grasp_ok & (~env._prev_release_after_grasp_grasp_ok)
    env._prev_release_after_grasp_grasp_ok = grasp_ok

    env._release_after_grasp_grace = torch.where(
        newly_grasped,
        torch.full_like(env._release_after_grasp_grace, int(grace_steps)),
        torch.clamp(env._release_after_grasp_grace - 1, min=0),
    )

    if reset_mask is not None:
        env._release_after_grasp_counter = torch.where(
            reset_mask, torch.zeros_like(env._release_after_grasp_counter), env._release_after_grasp_counter
        )
        env._release_after_grasp_grace = torch.where(
            reset_mask, torch.zeros_like(env._release_after_grasp_grace), env._release_after_grasp_grace
        )
        env._prev_release_after_grasp_grasp_ok = torch.where(
            reset_mask, torch.zeros_like(env._prev_release_after_grasp_grasp_ok), env._prev_release_after_grasp_grasp_ok
        )

    in_window = grasp_ok & (~physical_unlocked)
    bad = in_window & (env._release_after_grasp_grace == 0) & (~keep_ok)
    env._release_after_grasp_counter = torch.where(
        bad,
        env._release_after_grasp_counter + 1,
        torch.zeros_like(env._release_after_grasp_counter),
    )
    done = env._release_after_grasp_counter >= int(max(1, fail_steps))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["termination/release_after_grasp_bad_ratio"] = bad.float().mean().detach()
    env.extras["log"]["termination/release_after_grasp_counter_mean"] = (
        env._release_after_grasp_counter.float().mean().detach()
    )
    env.extras["log"]["termination/release_after_grasp_done_ratio"] = done.float().mean().detach()
    env.extras["log"]["termination/release_after_grasp_keep_ok_ratio"] = keep_ok.float().mean().detach()
    env.extras["log"]["termination/release_after_grasp_grace_mean"] = (
        env._release_after_grasp_grace.float().mean().detach()
    )

    return done
