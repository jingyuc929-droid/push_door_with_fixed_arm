from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _physical_door_unlocked(env: "ManagerBasedRLEnv") -> torch.Tensor:
    if hasattr(env, "_door_lock_mode"):
        return env._door_lock_mode == 2
    if hasattr(env, "_door_unlocked"):
        return env._door_unlocked.to(dtype=torch.bool)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


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
    """Rotate vector(s) v by quaternion(s) q (wxyz)."""
    qv = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return quat_mul(quat_mul(q, qv), quat_conjugate(q))[..., 1:]


def _body_pose_w(env: "ManagerBasedRLEnv", cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    asset: Articulation = env.scene[cfg.name]
    bid = cfg.body_ids[0]
    pos = asset.data.body_pos_w[:, bid, :]
    quat = asset.data.body_quat_w[:, bid, :]  # wxyz
    return pos, quat


# -----------------------------------------------------------------------------
# Geometry primitives
# -----------------------------------------------------------------------------


def fingertip_mid_to_handle_distance(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """|| (pL+pR)/2 - pHandle ||."""
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pTip = 0.5 * (pL + pR)
    pH, _ = _body_pose_w(env, handle_cfg)
    return torch.linalg.norm(pTip - pH, dim=-1)


def fingertip_mid_to_handle_grasp_point_distance(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    handle_offset_h: tuple[float, float, float] =  (-0.08, 0.04, 0.01),  # 在 handle frame 下的偏移
) -> torch.Tensor:
    # fingertips mid in world
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pTip = 0.5 * (pL + pR)

    # handle pose in world
    pH, qH = _body_pose_w(env, handle_cfg)

    # offset point in world: pG = pH + R_handle * offset_h
    off = torch.tensor(handle_offset_h, device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)
    pG = pH + quat_rotate(qH, off)

    return torch.linalg.norm(pTip - pG, dim=-1)


def ee_tcp_to_handle_grasp_point_distance(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, 0.01),
) -> torch.Tensor:
    pHand, qHand = _body_pose_w(env, hand_cfg)
    pH, qH = _body_pose_w(env, handle_cfg)

    ee_off = torch.tensor(ee_offset_pos, device=pHand.device, dtype=pHand.dtype).unsqueeze(0).repeat(pHand.shape[0], 1)
    h_off = torch.tensor(handle_offset_h, device=pH.device, dtype=pH.dtype).unsqueeze(0).repeat(pH.shape[0], 1)

    pTcp = pHand + quat_rotate(qHand, ee_off)
    pG = pH + quat_rotate(qH, h_off)
    return torch.linalg.norm(pTcp - pG, dim=-1)


def _gripper_width_and_closedness(
    env: "ManagerBasedRLEnv",
    gripper_cfg: SceneEntityCfg,
    open_width: float = 0.09,
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[gripper_cfg.name]
    jids = gripper_cfg.joint_ids
    q = robot.data.joint_pos[:, jids]

    if q.ndim == 2 and q.shape[1] >= 2:
        width = q.sum(dim=-1)
    else:
        width = 2.0 * q.reshape(-1)

    closedness = 1.0 - torch.clamp(width / open_width, 0.0, 1.0)
    return width, closedness


def _active_mean(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if torch.any(active):
        return value[active].mean().detach()
    return torch.zeros((), dtype=value.dtype, device=value.device)


def _active_max(value: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    if torch.any(active):
        return value[active].max().detach()
    return torch.zeros((), dtype=value.dtype, device=value.device)


# -----------------------------------------------------------------------------
# Handle-only contact force (filtered) utilities
# Requires ContactSensorCfg(filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/handle_1"])
# -----------------------------------------------------------------------------
def filtered_contact_force_vec_w(sensor: ContactSensor) -> torch.Tensor:
    """Filtered contact force vector in world frame, per env: [N,3].

    Uses sensor.data.force_matrix_w. We DO NOT fall back to net_forces_w to avoid self-collision noise.
    """
    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is None:
        return torch.zeros((sensor.data.net_forces_w.shape[0], 3), device=sensor.data.net_forces_w.device)

    # shapes commonly:
    #  [N, B, K, 3]  (B tracked bodies; K filtered prims)
    #  [N, K, 3]
    if fm.ndim == 4:
        vec = fm.sum(dim=2)  # [N, B, 3]
        # select the body with max norm (B usually 1)
        mag = torch.linalg.norm(vec, dim=-1)  # [N, B]
        idx = torch.argmax(mag, dim=1)
        return vec[torch.arange(vec.shape[0], device=vec.device), idx, :]
    if fm.ndim == 3:
        vec = fm.sum(dim=1)  # [N, 3]
        return vec

    raise RuntimeError(f"Unexpected force_matrix_w shape: {tuple(fm.shape)}")


#-----------------align grasp helpers-----------------
def filtered_contact_force_norm(sensor: ContactSensor) -> torch.Tensor:
    v = filtered_contact_force_vec_w(sensor)
    return torch.linalg.norm(v, dim=-1)


def compute_stage1_grasp_quality(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, 0.01),
    near_sigma: float = 0.06,
    near_hard_threshold: float = 0.14,
    grasp_axis: int = 2,
    min_sep: float = 0.005,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    contact_threshold: float = 0.25,
    contact_scale: float = 0.50,
    balance_power: float = 0.5,
    open_width: float = 0.09,
    min_closedness: float = 0.35,
    target_closedness: float = 0.65,
    max_closedness: float = 0.98,
    single_force_high: float = 1.0,
    single_force_low: float = 0.15,
) -> dict[str, torch.Tensor]:
    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    physical_unlocked = _physical_door_unlocked(env)
    active = grasp_ok & (~physical_unlocked)

    p_hand, q_hand = _body_pose_w(env, hand_cfg)
    p_handle, q_handle = _body_pose_w(env, handle_cfg)
    p_left, _ = _body_pose_w(env, left_finger_cfg)
    p_right, _ = _body_pose_w(env, right_finger_cfg)

    ee_offset = torch.tensor(ee_offset_pos, device=p_hand.device, dtype=p_hand.dtype).unsqueeze(0).repeat(p_hand.shape[0], 1)
    handle_offset = torch.tensor(handle_offset_h, device=p_handle.device, dtype=p_handle.dtype).unsqueeze(0).repeat(p_handle.shape[0], 1)
    p_tcp = p_hand + quat_rotate(q_hand, ee_offset)
    p_target = p_handle + quat_rotate(q_handle, handle_offset)
    tcp_dist = torch.linalg.norm(p_tcp - p_target, dim=-1)

    near_score = torch.exp(-torch.square(tcp_dist / float(max(near_sigma, 1e-6))))
    near_score = torch.where(tcp_dist <= float(near_hard_threshold), near_score, torch.zeros_like(near_score))

    q_handle_inv = quat_conjugate(q_handle)
    left_h = quat_rotate(q_handle_inv, p_left - p_handle)
    right_h = quat_rotate(q_handle_inv, p_right - p_handle)
    l = left_h[:, grasp_axis]
    r = right_h[:, grasp_axis]

    opposite = (l * r < 0.0).float()
    sep = torch.abs(l - r)
    sep_score = torch.tanh(torch.clamp(sep - float(min_sep), min=0.0) / float(max(sep_scale, 1e-6)))
    symmetry_error = torch.abs(torch.abs(l) - torch.abs(r))
    symmetry_score = torch.exp(-symmetry_error / float(max(symmetry_scale, 1e-6)))
    wrap_score = (opposite * sep_score * symmetry_score).clamp(0.0, 1.0)

    gripper_open_axis_w = _safe_unit(_local_vec_to_world(q_hand, gripper_open_axis_hand))
    gripper_approach_axis_w = _safe_unit(_local_vec_to_world(q_hand, gripper_approach_axis_hand))
    handle_grasp_axis_w = _safe_unit(_handle_axis_world(q_handle, grasp_axis))
    handle_approach_axis_w = _safe_unit(_handle_axis_world(q_handle, handle_approach_axis))

    open_dot = torch.sum(gripper_open_axis_w * handle_grasp_axis_w, dim=-1)
    open_align = torch.abs(open_dot).clamp(0.0, 1.0)
    approach_dot_raw = torch.sum(gripper_approach_axis_w * handle_approach_axis_w, dim=-1)
    approach_align = (float(expected_approach_sign) * approach_dot_raw).clamp(0.0, 1.0)
    pose_score = (0.40 * wrap_score + 0.20 * open_align + 0.40 * approach_align).clamp(0.0, 1.0)

    f_left = filtered_contact_force_norm(env.scene[left_sensor_name])
    f_right = filtered_contact_force_norm(env.scene[right_sensor_name])
    f_min = torch.minimum(f_left, f_right)
    f_max = torch.maximum(f_left, f_right)
    contact_ok = f_min > float(contact_threshold)
    contact_score = torch.tanh(
        torch.clamp(f_min - float(contact_threshold), min=0.0) / float(max(contact_scale, 1e-6))
    )
    balance_score = (f_min / (f_max + 1.0e-6)).clamp(0.0, 1.0) ** float(balance_power)

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_low = (
        (closedness - float(min_closedness)) / float(max(target_closedness - min_closedness, 1e-6))
    ).clamp(0.0, 1.0)
    close_high = (
        (float(max_closedness) - closedness) / float(max(max_closedness - target_closedness, 1e-6))
    ).clamp(0.0, 1.0)
    closure_score = close_low * close_high

    geometry_score = (
        0.25 * near_score
        + 0.35 * wrap_score
        + 0.25 * pose_score
        + 0.15 * closure_score
    )
    contact_quality = contact_score * (0.5 + 0.5 * balance_score)
    quality = (geometry_score * contact_quality).clamp(0.0, 1.0)

    closed_no_contact = (closedness > 0.75) & (~contact_ok)
    single_finger = (f_max > float(single_force_high)) & (f_min < float(single_force_low))

    return {
        "active": active,
        "quality": quality,
        "near_score": near_score,
        "wrap_score": wrap_score,
        "open_align": open_align,
        "approach_dot_raw": approach_dot_raw,
        "approach_align": approach_align,
        "pose_score": pose_score,
        "contact_score": contact_score,
        "contact_ok": contact_ok,
        "balance_score": balance_score,
        "closure_score": closure_score,
        "closedness": closedness,
        "tcp_dist": tcp_dist,
        "f_left": f_left,
        "f_right": f_right,
        "f_min": f_min,
        "f_max": f_max,
        "closed_no_contact": closed_no_contact,
        "single_finger": single_finger,
    }


def grasp_quality_keep_reward(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, 0.01),
    near_sigma: float = 0.06,
    near_hard_threshold: float = 0.14,
    grasp_axis: int = 2,
    min_sep: float = 0.005,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    expected_approach_sign: float = 1.0,
    contact_threshold: float = 0.25,
    contact_scale: float = 0.50,
    balance_power: float = 0.5,
    open_width: float = 0.09,
    min_closedness: float = 0.35,
    target_closedness: float = 0.65,
    max_closedness: float = 0.98,
    single_force_high: float = 1.0,
    single_force_low: float = 0.15,
    closed_no_contact_penalty: float = 0.5,
    single_finger_penalty: float = 0.5,
) -> torch.Tensor:
    terms = compute_stage1_grasp_quality(
        env=env,
        hand_cfg=hand_cfg,
        handle_cfg=handle_cfg,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        gripper_cfg=gripper_cfg,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
        near_sigma=near_sigma,
        near_hard_threshold=near_hard_threshold,
        grasp_axis=grasp_axis,
        min_sep=min_sep,
        sep_scale=sep_scale,
        symmetry_scale=symmetry_scale,
        gripper_open_axis_hand=gripper_open_axis_hand,
        gripper_approach_axis_hand=gripper_approach_axis_hand,
        handle_approach_axis=handle_approach_axis,
        expected_approach_sign=expected_approach_sign,
        contact_threshold=contact_threshold,
        contact_scale=contact_scale,
        balance_power=balance_power,
        open_width=open_width,
        min_closedness=min_closedness,
        target_closedness=target_closedness,
        max_closedness=max_closedness,
        single_force_high=single_force_high,
        single_force_low=single_force_low,
    )
    reward = (
        terms["quality"]
        - float(closed_no_contact_penalty) * terms["closed_no_contact"].float()
        - float(single_finger_penalty) * terms["single_finger"].float()
    )
    return terms["active"].float() * reward


def _local_vec_to_world(q: torch.Tensor, v_local: tuple[float, float, float]) -> torch.Tensor:
    v = torch.tensor(v_local, device=q.device, dtype=q.dtype).unsqueeze(0).repeat(q.shape[0], 1)
    return quat_rotate(q, v)


def _axis_idx_to_local_vec(axis_idx: int, device, dtype) -> torch.Tensor:
    e = torch.zeros((1, 3), device=device, dtype=dtype)
    e[0, int(axis_idx)] = 1.0
    return e


def _handle_axis_world(qH: torch.Tensor, axis_idx: int) -> torch.Tensor:
    e = _axis_idx_to_local_vec(axis_idx, qH.device, qH.dtype).repeat(qH.shape[0], 1)
    return quat_rotate(qH, e)


def _safe_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


# -----------------------------------------------------------------------------
# ----------------Helper-------------------------------------------------------
# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# (1) Approach + (2) Wrap alignment
# -----------------------------------------------------------------------------
def approach_handle_inv_square(
    env: "ManagerBasedRLEnv",
    hand_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h =  (-0.08, 0.04, 0.01),
    eps: float = 1e-4,
    scale: float = 0.10,
    clip: float = 1.0,
) -> torch.Tensor:
    d = ee_tcp_to_handle_grasp_point_distance(
        env,
        hand_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )
    r = scale / (d * d + eps)
    return torch.clamp(r, max=clip)

def align_grasp_around_handle_local(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    grasp_axis: int = 2,   # confirmed: handle-frame z
    min_sep: float = 0.002,
) -> torch.Tensor:
    """1 if fingertips are on opposite sides along handle-frame grasp_axis (z)."""
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pH, qH = _body_pose_w(env, handle_cfg)

    L_h = quat_rotate(quat_conjugate(qH), pL - pH)  # [N,3]
    R_h = quat_rotate(quat_conjugate(qH), pR - pH)

    side = (L_h[:, grasp_axis] * R_h[:, grasp_axis]) < 0.0
    sep = torch.abs(L_h[:, grasp_axis] - R_h[:, grasp_axis]) > min_sep
    return (side & sep).float()

def align_grasp_pose_v2_terms(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    side_weight: float = 0.55,
    open_weight: float = 0.25,
    approach_weight: float = 0.20,
):
    pL, _ = _body_pose_w(env, left_finger_cfg)
    pR, _ = _body_pose_w(env, right_finger_cfg)
    pH, qH = _body_pose_w(env, handle_cfg)
    _, qG = _body_pose_w(env, hand_cfg)

    qHc = quat_conjugate(qH)
    L_h = quat_rotate(qHc, pL - pH)
    R_h = quat_rotate(qHc, pR - pH)

    l = L_h[:, grasp_axis]
    r = R_h[:, grasp_axis]

    opposite = (l * r < 0.0).float()
    sep = torch.abs(l - r)
    sep_score = torch.tanh(torch.clamp(sep - float(min_sep), min=0.0) / float(sep_scale))

    sym_err = torch.abs(torch.abs(l) - torch.abs(r))
    sym_score = torch.exp(-sym_err / float(symmetry_scale))

    side_score = opposite * sep_score * sym_score

    g_open_w = _safe_unit(_local_vec_to_world(qG, gripper_open_axis_hand))
    h_grasp_w = _safe_unit(_handle_axis_world(qH, grasp_axis))
    open_dot = torch.sum(g_open_w * h_grasp_w, dim=-1)
    open_align = torch.abs(open_dot).clamp(0.0, 1.0)

    g_app_w = _safe_unit(_local_vec_to_world(qG, gripper_approach_axis_hand))
    h_app_w = _safe_unit(_handle_axis_world(qH, handle_approach_axis))
    app_dot = torch.sum(g_app_w * h_app_w, dim=-1)
    approach_align = torch.abs(app_dot).clamp(0.0, 1.0)

    score_open = side_score * open_align
    score_app = side_score * approach_align

    wsum = float(side_weight + open_weight + approach_weight)
    score = (
        float(side_weight) * side_score
        + float(open_weight) * score_open
        + float(approach_weight) * score_app
    ) / max(wsum, 1e-6)
    score = score.clamp(0.0, 1.0)

    return {
        "score": score,
        "side_score": side_score,
        "open_align": open_align,
        "approach_align": approach_align,
        "open_dot": open_dot,
        "approach_dot": app_dot,
        "l": l,
        "r": r,
        "sep": sep,
        "sym_err": sym_err,
    }

def align_grasp_pose_v2(
    env: "ManagerBasedRLEnv",
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    side_weight: float = 0.55,
    open_weight: float = 0.25,
    approach_weight: float = 0.20,
) -> torch.Tensor:
    terms = align_grasp_pose_v2_terms(
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
        side_weight=side_weight,
        open_weight=open_weight,
        approach_weight=approach_weight,
    )
    return terms["score"]


# -----------------------------------------------------------------------------
# (3) Close shaping when ready (pre-grasp)
# -----------------------------------------------------------------------------
def close_gripper_shaping_when_ready(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    distance_threshold: float = 0.10,
    ee_offset_pos: tuple[float, float, float] = (0.1523, 0.0, 0.0),
    handle_offset_h =  (-0.09, 0.04, 0.01),
    # --- v2 align gate ---
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    side_weight: float = 0.55,
    open_weight: float = 0.25,
    approach_weight: float = 0.20,
    align_threshold: float = 0.20,
    # --- contact/closure ---
    require_any_contact: bool = True,
    contact_threshold: float = 0.5,
    open_width: float = 0.09,
) -> torch.Tensor:
    dist = ee_tcp_to_handle_grasp_point_distance(
        env,
        hand_cfg,
        handle_cfg,
        ee_offset_pos=ee_offset_pos,
        handle_offset_h=handle_offset_h,
    )
    near_ok = dist < distance_threshold

    align = align_grasp_pose_v2(
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
        side_weight=side_weight,
        open_weight=open_weight,
        approach_weight=approach_weight,
    )
    align_ok = align >= align_threshold

    gate = near_ok & align_ok
    if require_any_contact:
        fL = filtered_contact_force_norm(env.scene[left_sensor_name])
        fR = filtered_contact_force_norm(env.scene[right_sensor_name])
        gate = gate & (torch.maximum(fL, fR) > contact_threshold)

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    return gate.float() * closedness


# -----------------------------------------------------------------------------
# (4) Grasp reward: direction-specific force in handle frame (|Fz|) + side penalty
# -----------------------------------------------------------------------------
def grasp_handle_reward(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # gates
    distance_threshold: float = 0.08,
    handle_offset_h =  (-0.09, 0.04, 0.01),
    grasp_axis: int = 2,
    min_sep: float = 0.0050,
    sep_scale: float = 0.0010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    align_side_weight: float = 0.55,
    align_open_weight: float = 0.25,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,
    # force shaping
    force_threshold: float = 2.0,
    force_scale: float = 10.0,
    side_scale: float = 10.0,
    side_weight: float = 0.3,
    # closure shaping
    open_width: float = 0.09,
    min_closedness: float = 0.4,
    close_scale: float = 1.0,
    close_power: float = 2.0,
    # anti-hacking
    hold_steps: int = 6,
    balance_eps: float = 1e-6,
    balance_power: float = 2.0,
    finger_speed_std: float = 0.05,
    hold_tau: float | None = None,
    hold_power: float = 1.0,
    hold_decay: float = 0.0,
) -> torch.Tensor:
    dist = fingertip_mid_to_handle_grasp_point_distance(
        env, left_finger_cfg, right_finger_cfg, handle_cfg, handle_offset_h=(handle_offset_h)
    )
    near_ok = dist < distance_threshold

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

    _, qH = _body_pose_w(env, handle_cfg)
    qHc = quat_conjugate(qH)

    left: ContactSensor = env.scene[left_sensor_name]
    right: ContactSensor = env.scene[right_sensor_name]
    FL_w = filtered_contact_force_vec_w(left)
    FR_w = filtered_contact_force_vec_w(right)

    FL_h = quat_rotate(qHc, FL_w)
    FR_h = quat_rotate(qHc, FR_w)

    Fz_L = torch.abs(FL_h[:, grasp_axis])
    Fz_R = torch.abs(FR_h[:, grasp_axis])
    Fz_min = torch.minimum(Fz_L, Fz_R)
    Fz_max = torch.maximum(Fz_L, Fz_R)

    contact_ok = Fz_min > force_threshold

    side_L = torch.abs(FL_h[:, 0]) + torch.abs(FL_h[:, 1])
    side_R = torch.abs(FR_h[:, 0]) + torch.abs(FR_h[:, 1])
    side_sum = side_L + side_R

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_ok = closedness > min_closedness

    gate = near_ok & wrap_ok & contact_ok & close_ok

    N = dist.shape[0]
    device = dist.device
    if (not hasattr(env, "_grasp_hold_counter")) or (env._grasp_hold_counter.shape[0] != N):
        env._grasp_hold_counter = torch.zeros(N, device=device, dtype=torch.int32)
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf == 0
        env._grasp_hold_counter = torch.where(reset_mask, torch.zeros_like(env._grasp_hold_counter), env._grasp_hold_counter)

    if hold_decay > 0.0:
        decayed = (env._grasp_hold_counter.float() * float(hold_decay)).to(torch.int32)
        env._grasp_hold_counter = torch.where(gate, env._grasp_hold_counter + 1, decayed)
    else:
        env._grasp_hold_counter = torch.where(gate, env._grasp_hold_counter + 1, torch.zeros_like(env._grasp_hold_counter))

    tau = float(hold_tau) if hold_tau is not None else float(max(1.0, hold_steps / 2.0))
    hold_prog = 1.0 - torch.exp(-env._grasp_hold_counter.float() / tau)
    hold_prog = torch.clamp(hold_prog, 0.0, 1.0) ** float(hold_power)

    balance = (Fz_min / (Fz_max + balance_eps)).clamp(0.0, 1.0) ** balance_power
    r_main = torch.tanh(torch.clamp(Fz_min - force_threshold, min=0.0) / force_scale)
    r_side = torch.tanh(side_sum / side_scale)
    close_factor = (closedness.clamp(0.0, 1.0) ** close_power)

    robot: Articulation = env.scene[gripper_cfg.name]
    dq = torch.abs(robot.data.joint_vel[:, gripper_cfg.joint_ids]).sum(dim=-1)
    stability = 0.3 + 0.7 * torch.exp(-dq / finger_speed_std)

    rew = hold_prog * balance * stability
    rew = rew * (r_main - side_weight * r_side).clamp(min=0.0)
    rew = rew * (0.1 + 0.9 * (1.0 + close_scale * close_factor))
    return rew

def grasp_handle_reward_preunlock_only(
    env: "ManagerBasedRLEnv",
    # new control params
    handle_joint_cfg: SceneEntityCfg,
    unlock_enter_pos: float = -0.03,
    fade_width: float = 0.02,
    less_than: bool = True,

    # --- original grasp_handle_reward params ---
    handle_cfg: SceneEntityCfg = SceneEntityCfg("door", body_names=["handle_1"]),
    handle_offset_h: tuple[float, float, float] = (-0.09, 0.04, 0.01),
    left_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link7"]),
    right_finger_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link8"]),
    hand_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link6"]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper_joint"]),
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",

    distance_threshold: float = 0.10,
    grasp_axis: int = 2,
    min_sep: float = 0.005,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    align_side_weight: float = 0.70,
    align_open_weight: float = 0.10,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,

    force_threshold: float = 0.5,
    force_scale: float = 10.0,
    side_scale: float = 10.0,
    side_weight: float = 0.3,

    open_width: float = 0.09,
    min_closedness: float = 0.3,
    close_scale: float = 1.0,
    close_power: float = 2.0,

    hold_steps: int = 2,
    hold_tau: float = 2.5,
    hold_power: float = 1.0,
    hold_decay: float = 0.6,
    balance_power: float = 2.0,
    finger_speed_std: float = 0.08,
) -> torch.Tensor:
    """
    Use grasp_handle_reward only before the press/unlock phase.
    Once handle_pos goes sufficiently into press direction, fade this reward out.
    """

    base = grasp_handle_reward(
        env=env,
        handle_cfg=handle_cfg,
        handle_offset_h=handle_offset_h,
        left_finger_cfg=left_finger_cfg,
        right_finger_cfg=right_finger_cfg,
        hand_cfg=hand_cfg,
        gripper_cfg=gripper_cfg,
        left_sensor_name=left_sensor_name,
        right_sensor_name=right_sensor_name,
        distance_threshold=distance_threshold,
        grasp_axis=grasp_axis,
        min_sep=min_sep,
        sep_scale=sep_scale,
        symmetry_scale=symmetry_scale,
        gripper_open_axis_hand=gripper_open_axis_hand,
        gripper_approach_axis_hand=gripper_approach_axis_hand,
        handle_approach_axis=handle_approach_axis,
        align_side_weight=align_side_weight,
        align_open_weight=align_open_weight,
        align_approach_weight=align_approach_weight,
        align_threshold=align_threshold,
        force_threshold=force_threshold,
        force_scale=force_scale,
        side_scale=side_scale,
        side_weight=side_weight,
        open_width=open_width,
        min_closedness=min_closedness,
        close_scale=close_scale,
        close_power=close_power,
        hold_steps=hold_steps,
        hold_tau=hold_tau,
        hold_power=hold_power,
        hold_decay=hold_decay,
        balance_power=balance_power,
        finger_speed_std=finger_speed_std,
    )

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )

    if less_than:
        # e.g. unlock_enter_pos=-0.03, fade_width=0.02:
        # handle_pos >= -0.03 => scale=1
        # handle_pos <= -0.05 => scale=0
        fade_end = float(unlock_enter_pos) - float(fade_width)
        scale = torch.clamp(
            (handle_pos - fade_end) / float(max(fade_width, 1e-6)),
            0.0, 1.0
        )
    else:
        fade_end = float(unlock_enter_pos) + float(fade_width)
        scale = torch.clamp(
            (fade_end - handle_pos) / float(max(fade_width, 1e-6)),
            0.0, 1.0
        )

    return base * scale


# -----------------------------------------------------------------------------
# (5) Sparse success bonus (one-shot per episode)
# -----------------------------------------------------------------------------
def grasp_success_bonus(
    env: "ManagerBasedRLEnv",
    handle_cfg: SceneEntityCfg,
    left_finger_cfg: SceneEntityCfg,
    right_finger_cfg: SceneEntityCfg,
    hand_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    distance_threshold: float = 0.1,
    force_threshold: float = 1.0,
    hold_steps: int = 5,
    open_width: float = 0.09,
    min_closedness: float = 0.5,
    bonus: float = 20.0,
    require_wrap: bool = True,
    require_any_finger_contact: bool = False,
    use_force_norm: bool = True,
    near_mode: str = "min_finger",
    use_grasp_point: bool = True,
    handle_offset_h: tuple[float, float, float] =  (-0.09, 0.04, 0.01),
    archive_cap: int = 512,
    handle_joint_cfg: SceneEntityCfg | None = None,
    relax_near_after_handle_pos: float = -0.05,
    less_than: bool = True,
    # --- v2 wrap gate ---
    grasp_axis: int = 2,
    min_sep: float = 0.010,
    sep_scale: float = 0.010,
    symmetry_scale: float = 0.015,
    gripper_open_axis_hand: tuple[float, float, float] = (0.0, 1.0, 0.0),
    gripper_approach_axis_hand: tuple[float, float, float] = (1.0, 0.0, 0.0),
    handle_approach_axis: int = 1,
    align_side_weight: float = 0.55,
    align_open_weight: float = 0.25,
    align_approach_weight: float = 0.20,
    align_threshold: float = 0.30,
) -> torch.Tensor:
    if use_grasp_point:
        dist = fingertip_mid_to_handle_grasp_point_distance(
            env, left_finger_cfg, right_finger_cfg, handle_cfg, handle_offset_h=handle_offset_h
        )
    else:
        dist = fingertip_mid_to_handle_distance(env, left_finger_cfg, right_finger_cfg, handle_cfg)

    near_ok = dist < distance_threshold

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

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_ok = closedness > min_closedness

    sL = env.scene[left_sensor_name]
    sR = env.scene[right_sensor_name]

    if use_force_norm:
        fL = filtered_contact_force_norm(sL)
        fR = filtered_contact_force_norm(sR)
    else:
        _, qH2 = _body_pose_w(env, handle_cfg)
        qHc = quat_conjugate(qH2)
        FL_h = quat_rotate(qHc, filtered_contact_force_vec_w(sL))
        FR_h = quat_rotate(qHc, filtered_contact_force_vec_w(sR))
        fL = torch.abs(FL_h[:, grasp_axis])
        fR = torch.abs(FR_h[:, grasp_axis])

    if require_any_finger_contact:
        contact_ok = torch.maximum(fL, fR) > force_threshold
    else:
        contact_ok = torch.minimum(fL, fR) > force_threshold

    if handle_joint_cfg is not None:
        door: Articulation = env.scene[handle_joint_cfg.name]
        jids = handle_joint_cfg.joint_ids
        handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

        if less_than:
            relax_phase = handle_pos < float(relax_near_after_handle_pos)
        else:
            relax_phase = handle_pos > float(relax_near_after_handle_pos)
    else:
        handle_pos = torch.zeros_like(dist)
        relax_phase = torch.zeros_like(near_ok, dtype=torch.bool)

    gate_strict = near_ok & wrap_ok & close_ok & contact_ok
    gate_relaxed = close_ok & contact_ok
    gate = torch.where(relax_phase, gate_relaxed, gate_strict)

    N = dist.shape[0]
    device = dist.device
    if (not hasattr(env, "_grasp_success_counter")) or (env._grasp_success_counter.shape[0] != N):
        env._grasp_success_counter = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_grasp_success_given")) or (env._grasp_success_given.shape[0] != N):
        env._grasp_success_given = torch.zeros(N, device=device, dtype=torch.bool)

    if not hasattr(env, "_grasp_success_init_cleared"):
        env._grasp_success_counter.zero_()
        env._grasp_success_given.zero_()
        env._grasp_success_init_cleared = True

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_grasp_success")) or (env._prev_ep_len_grasp_success.shape[0] != N):
            env._prev_ep_len_grasp_success = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_grasp_success) | (env.episode_length_buf == 0)
        env._prev_ep_len_grasp_success = env.episode_length_buf.clone()

    if reset_mask is not None:
        env._grasp_success_counter = torch.where(reset_mask, torch.zeros_like(env._grasp_success_counter), env._grasp_success_counter)
        env._grasp_success_given = torch.where(reset_mask, torch.zeros_like(env._grasp_success_given), env._grasp_success_given)

    env._grasp_success_counter = torch.where(gate, env._grasp_success_counter + 1, torch.zeros_like(env._grasp_success_counter))
    success_now = env._grasp_success_counter >= int(max(1, hold_steps))

    give = success_now & (~env._grasp_success_given)
    env._grasp_success_given = env._grasp_success_given | give

    idx = torch.nonzero(give).squeeze(-1)
    if idx.numel() > 0:
        from .stage import push_archive_from_env
        push_archive_from_env(
            env=env,
            name="_archive_grasp",
            env_ids=idx,
            cap=archive_cap,
            robot_cfg=SceneEntityCfg("robot"),
            door_cfg=SceneEntityCfg("door"),
            store_unlock_flag=False,
        )

    return give.float() * float(bonus)

# ---------------------encourage press handle---------------
def press_handle_after_grasp_vel(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    # 轻 gate：避免没抓住乱按
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.5,
    require_any_contact: bool = True,
    open_width: float = 0.08,
    min_closedness: float = 0.35,
    # 方向/抗抖
    less_than: bool = True,
    vel_deadzone: float = 0.01,
    vel_scale: float = 0.05,
    opposite_penalty: float = 0.2,
    clip: float = 1.0,
    # --- NEW: anti-spike by requiring net position progress ---
    pos_deadzone: float = 1e-4,      # 忽略极小位移（数值噪声）
    pos_scale: float = 2e-3,         # Δpos -> [0,1] 的平滑尺度（越小越“硬门控”）
    use_vel_ema: bool = True,
    vel_ema_alpha: float = 0.25,     # EMA 平滑系数（0~1）
) -> torch.Tensor:
    # 只在本 episode 已经 grasp_success 后激活
    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(env.num_envs, device=env.device)
    phase = env._grasp_success_given  # [N] bool

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    if len(jids) == 1:
        jid = jids[0]
        handle_pos = door.data.joint_pos[:, jid]
        handle_vel_raw = door.data.joint_vel[:, jid]
    else:
        handle_pos = door.data.joint_pos[:, jids].mean(dim=-1)
        handle_vel_raw = door.data.joint_vel[:, jids].mean(dim=-1)

    # light gate (same style as unlock)
    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (torch.maximum(fL, fR) > contact_threshold) if require_any_contact else (torch.minimum(fL, fR) > contact_threshold)
    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_ok = closedness > min_closedness
    gate = contact_ok & close_ok

    # --- robust reset detection for buffers ---
    N = handle_pos.shape[0]
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_press")) or (env._prev_ep_len_press.shape[0] != N):
            env._prev_ep_len_press = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_press) | (env.episode_length_buf == 0)
        env._prev_ep_len_press = env.episode_length_buf.clone()

    # --- cache prev handle_pos to measure net progress ---
    if (not hasattr(env, "_press_prev_handle_pos")) or (env._press_prev_handle_pos.shape[0] != N):
        env._press_prev_handle_pos = handle_pos.clone()
    if reset_mask is not None:
        env._press_prev_handle_pos = torch.where(reset_mask, handle_pos, env._press_prev_handle_pos)

    # Δpos in desired direction (positive means progressing toward unlock)
    # less_than=True means handle_pos should decrease (become more negative)
    dpos = (env._press_prev_handle_pos - handle_pos) if less_than else (handle_pos - env._press_prev_handle_pos)
    env._press_prev_handle_pos = handle_pos.clone()

    # smooth progress gate in [0,1] (0 if no net progress)
    prog_gate = torch.tanh(torch.clamp(dpos - float(pos_deadzone), min=0.0) / float(pos_scale))

    # --- EMA smooth velocity to reduce impulsive spikes ---
    if use_vel_ema:
        if (not hasattr(env, "_press_vel_ema")) or (env._press_vel_ema.shape[0] != N):
            env._press_vel_ema = torch.zeros_like(handle_vel_raw)
        if reset_mask is not None:
            env._press_vel_ema = torch.where(reset_mask, torch.zeros_like(env._press_vel_ema), env._press_vel_ema)
        env._press_vel_ema = (1.0 - float(vel_ema_alpha)) * env._press_vel_ema + float(vel_ema_alpha) * handle_vel_raw
        handle_vel = env._press_vel_ema
    else:
        handle_vel = handle_vel_raw

    # velocity shaping (same as before)
    desired = (-handle_vel) if less_than else (handle_vel)
    pos = torch.clamp(desired - float(vel_deadzone), min=0.0)
    neg = torch.clamp(-desired - float(vel_deadzone), min=0.0)

    # KEY FIX:
    #  - positive reward is multiplied by prog_gate (needs net position progress)
    #  - negative (wrong-direction) still penalized to discourage oscillation spikes
    r_pos = torch.tanh(pos / float(vel_scale)) * prog_gate
    r_neg = torch.tanh(neg / float(vel_scale))
    r = r_pos - float(opposite_penalty) * r_neg

    r = torch.clamp(r, -float(clip), float(clip))
    return (phase & gate).float() * r

#----------discourage local optimum -----------------------------
def stall_penalty_after_grasp(
    env: "ManagerBasedRLEnv",
    # --- use primitives only (Hydra-safe) ---
    door_asset_name: str = "door",
    handle_joint_name: str = "handle_joint",
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.5,
    require_any_contact: bool = True,
    # stall 判定
    vel_eps: float = 0.01,
    penalty: float = 0.01,
    # --- phase window ---
    recent_window_steps: int = 120,
    grace_steps: int = 10,
) -> torch.Tensor:
    # Need grasp_success signal (set by grasp_success_bonus)
    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(env.num_envs, device=env.device)

    N = env.num_envs
    device = env.device

    # ---------------- robust reset detection ----------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_stall")) or (env._prev_ep_len_stall.shape[0] != N):
            env._prev_ep_len_stall = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_stall) | (env.episode_length_buf == 0)
        env._prev_ep_len_stall = env.episode_length_buf.clone()

    # ---------------- phase TTL bookkeeping ----------------
    if (not hasattr(env, "_grasp_recent_ttl")) or (env._grasp_recent_ttl.shape[0] != N):
        env._grasp_recent_ttl = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_grasp_recent_grace")) or (env._grasp_recent_grace.shape[0] != N):
        env._grasp_recent_grace = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_prev_grasp_success_given")) or (env._prev_grasp_success_given.shape[0] != N):
        env._prev_grasp_success_given = env._grasp_success_given.clone()

    if reset_mask is not None:
        env._grasp_recent_ttl = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_ttl), env._grasp_recent_ttl)
        env._grasp_recent_grace = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_grace), env._grasp_recent_grace)
        env._prev_grasp_success_given = torch.where(reset_mask, torch.zeros_like(env._prev_grasp_success_given), env._prev_grasp_success_given)

    newly = env._grasp_success_given & (~env._prev_grasp_success_given)
    env._prev_grasp_success_given = env._grasp_success_given.clone()

    env._grasp_recent_ttl = torch.where(
        newly,
        torch.full_like(env._grasp_recent_ttl, int(recent_window_steps)),
        torch.clamp(env._grasp_recent_ttl - 1, min=0),
    )
    env._grasp_recent_grace = torch.where(
        newly,
        torch.full_like(env._grasp_recent_grace, int(grace_steps)),
        torch.clamp(env._grasp_recent_grace - 1, min=0),
    )

    phase = env._grasp_recent_ttl > 0
    grace_ok = env._grasp_recent_grace == 0

    # ---------------- contact gate (handle-only filtered) ----------------
    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (torch.maximum(fL, fR) > contact_threshold) if require_any_contact else (torch.minimum(fL, fR) > contact_threshold)

    # ---------------- handle joint velocity ----------------
    door: Articulation = env.scene[door_asset_name]

    # cache joint id lookup on env
    if not hasattr(env, "_cache_joint_id_map"):
        env._cache_joint_id_map = {}

    key = (door_asset_name, handle_joint_name)
    jid = env._cache_joint_id_map.get(key, None)
    if jid is None:
        # get joint names list
        if hasattr(door, "data") and hasattr(door.data, "joint_names"):
            jnames = list(door.data.joint_names)
        elif hasattr(door, "joint_names"):
            jnames = list(door.joint_names)
        else:
            raise RuntimeError("Cannot access door joint names to resolve handle_joint_name.")
        if handle_joint_name not in jnames:
            raise RuntimeError(f"Joint '{handle_joint_name}' not found in {door_asset_name}. Available: {jnames}")
        jid = int(jnames.index(handle_joint_name))
        env._cache_joint_id_map[key] = jid

    handle_vel = door.data.joint_vel[:, jid]
    stalled = torch.abs(handle_vel) < float(vel_eps)

    bad = phase & grace_ok & contact_ok & stalled
    return -float(penalty) * bad.float()

def stall_penalty_after_grasp_pos(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.2,
    require_any_contact: bool = False,
    stall_pos: float = -0.10,
    pos_scale: float = 0.03,
    penalty: float = 0.02,
    recent_window_steps: int = 200,
    grace_steps: int = 10,
    less_than: bool = True,
) -> torch.Tensor:
    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(env.num_envs, device=env.device)

    N = env.num_envs
    device = env.device

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_stall_pos")) or (env._prev_ep_len_stall_pos.shape[0] != N):
            env._prev_ep_len_stall_pos = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_stall_pos) | (env.episode_length_buf == 0)
        env._prev_ep_len_stall_pos = env.episode_length_buf.clone()

    if (not hasattr(env, "_grasp_recent_ttl")) or (env._grasp_recent_ttl.shape[0] != N):
        env._grasp_recent_ttl = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_grasp_recent_grace")) or (env._grasp_recent_grace.shape[0] != N):
        env._grasp_recent_grace = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_prev_grasp_success_given")) or (env._prev_grasp_success_given.shape[0] != N):
        env._prev_grasp_success_given = env._grasp_success_given.clone()

    if reset_mask is not None:
        env._grasp_recent_ttl = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_ttl), env._grasp_recent_ttl)
        env._grasp_recent_grace = torch.where(reset_mask, torch.zeros_like(env._grasp_recent_grace), env._grasp_recent_grace)
        env._prev_grasp_success_given = torch.where(reset_mask, torch.zeros_like(env._prev_grasp_success_given), env._prev_grasp_success_given)

    newly = env._grasp_success_given & (~env._prev_grasp_success_given)
    env._prev_grasp_success_given = env._grasp_success_given.clone()

    env._grasp_recent_ttl = torch.where(
        newly,
        torch.full_like(env._grasp_recent_ttl, int(recent_window_steps)),
        torch.clamp(env._grasp_recent_ttl - 1, min=0),
    )
    env._grasp_recent_grace = torch.where(
        newly,
        torch.full_like(env._grasp_recent_grace, int(grace_steps)),
        torch.clamp(env._grasp_recent_grace - 1, min=0),
    )

    phase = env._grasp_recent_ttl > 0
    grace_ok = env._grasp_recent_grace == 0

    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (torch.maximum(fL, fR) > contact_threshold) if require_any_contact else (torch.minimum(fL, fR) > contact_threshold)

    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    if less_than:
        bad_depth = torch.clamp((handle_pos - float(stall_pos)) / float(pos_scale), min=0.0)
    else:
        bad_depth = torch.clamp((float(stall_pos) - handle_pos) / float(pos_scale), min=0.0)

    bad = phase & grace_ok & contact_ok
    return -float(penalty) * bad.float() * torch.tanh(bad_depth)

def anti_release_after_press_to_open(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg | None = None,
    left_finger_cfg: SceneEntityCfg | None = None,
    right_finger_cfg: SceneEntityCfg | None = None,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # --- keep gate ---
    contact_threshold: float = 0.3,
    require_any_contact: bool = False,
    open_width: float = 0.09,
    min_closedness: float = 0.25,
    distance_threshold: float = 0.13,
    handle_offset_h: tuple[float, float, float] = (-0.08, 0.04, 0.01),
    # --- phase start ---
    handle_start_pos: float = 0.0,
    handle_threshold: float = -0.3,
    activate_progress: float = 0.25,
    use_unlock_success_latch: bool = True,
    # --- door open definition ---
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    push_enter_open: float = 0.02,
    door_open_threshold: float = 0.35,
    # --- transition window only ---
    max_keep_steps_after_unlock: int = 8,
    keep_until_door_open: bool = True,
    # --- reward scales ---
    hold_reward: float = 0.005,
    progress_boost: float = 0.03,
    release_event_penalty: float = 0.20,
    lost_penalty: float = 0.01,
    auto_open_penalty: float = 0.05,
) -> torch.Tensor:
    N = env.num_envs
    device = env.device

    if not hasattr(env, "_grasp_success_given"):
        return torch.zeros(N, device=device)

    # ------------------------------------------------------------
    # 1) keep gate = contact + close + near grasp point when geometry cfgs are provided
    # ------------------------------------------------------------
    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (torch.maximum(fL, fR) > contact_threshold) if require_any_contact else (torch.minimum(fL, fR) > contact_threshold)

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_ok = closedness > min_closedness
    if handle_cfg is not None and left_finger_cfg is not None and right_finger_cfg is not None:
        dist = fingertip_mid_to_handle_grasp_point_distance(
            env,
            left_finger_cfg,
            right_finger_cfg,
            handle_cfg,
            handle_offset_h=handle_offset_h,
        )
        near_ok = dist < float(distance_threshold)
    else:
        dist = torch.zeros(N, device=device)
        near_ok = torch.ones(N, dtype=torch.bool, device=device)
    keep_ok = contact_ok & close_ok & near_ok

    # ------------------------------------------------------------
    # 2) handle progress
    # ------------------------------------------------------------
    handle_asset: Articulation = env.scene[handle_joint_cfg.name]
    hjids = handle_joint_cfg.joint_ids
    handle_pos = handle_asset.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else handle_asset.data.joint_pos[:, hjids].mean(dim=-1)

    denom = float(handle_threshold - handle_start_pos)
    if abs(denom) < 1e-9:
        handle_prog = torch.zeros_like(handle_pos)
    else:
        handle_prog = (handle_pos - float(handle_start_pos)) / denom
        handle_prog = torch.clamp(handle_prog, 0.0, 1.0)

    # ------------------------------------------------------------
    # 3) door open
    # ------------------------------------------------------------
    door_asset: Articulation = env.scene[door_joint_cfg.name]
    djids = door_joint_cfg.joint_ids
    door_pos = door_asset.data.joint_pos[:, djids[0]] if len(djids) == 1 else door_asset.data.joint_pos[:, djids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    # ------------------------------------------------------------
    # 4) reset detection
    # ------------------------------------------------------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_keep_transition")) or (env._prev_ep_len_keep_transition.shape[0] != N):
            env._prev_ep_len_keep_transition = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_keep_transition) | (env.episode_length_buf == 0)
        env._prev_ep_len_keep_transition = env.episode_length_buf.clone()

    # ------------------------------------------------------------
    # 5) unlock/open phase latch
    # ------------------------------------------------------------
    unlock_latch = torch.zeros(N, dtype=torch.bool, device=device)
    if use_unlock_success_latch and hasattr(env, "_unlock_success_given"):
        unlock_latch = env._unlock_success_given

    phase_raw = env._grasp_success_given & (
        (handle_prog > float(activate_progress))
        | unlock_latch
        | (door_open > float(push_enter_open))
    )

    if (not hasattr(env, "_keep_transition_phase")) or (env._keep_transition_phase.shape[0] != N):
        env._keep_transition_phase = torch.zeros(N, dtype=torch.bool, device=device)
    if (not hasattr(env, "_keep_transition_ttl")) or (env._keep_transition_ttl.shape[0] != N):
        env._keep_transition_ttl = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_keep_transition_prev_keep_ok")) or (env._keep_transition_prev_keep_ok.shape[0] != N):
        env._keep_transition_prev_keep_ok = torch.zeros(N, dtype=torch.bool, device=device)

    if reset_mask is not None:
        env._keep_transition_phase = torch.where(reset_mask, torch.zeros_like(env._keep_transition_phase), env._keep_transition_phase)
        env._keep_transition_ttl = torch.where(reset_mask, torch.zeros_like(env._keep_transition_ttl), env._keep_transition_ttl)
        env._keep_transition_prev_keep_ok = torch.where(reset_mask, torch.zeros_like(env._keep_transition_prev_keep_ok), env._keep_transition_prev_keep_ok)

    newly = phase_raw & (~env._keep_transition_phase)
    env._keep_transition_phase = env._keep_transition_phase | phase_raw
    env._keep_transition_ttl = torch.where(
        newly,
        torch.full_like(env._keep_transition_ttl, int(max_keep_steps_after_unlock)),
        torch.clamp(env._keep_transition_ttl - 1, min=0),
    )

    phase = env._keep_transition_ttl > 0
    if keep_until_door_open:
        phase = phase & (door_open < float(door_open_threshold))

    # ------------------------------------------------------------
    # 6) release event inside short window only
    # ------------------------------------------------------------
    prev_keep = env._keep_transition_prev_keep_ok
    env._keep_transition_prev_keep_ok = phase & keep_ok

    release_event = phase & prev_keep & (~keep_ok)
    lost = phase & (~keep_ok)

    # ------------------------------------------------------------
    # 7) reward
    # ------------------------------------------------------------
    # only weak positive reward in the transition window
    hold_rew = (phase & keep_ok).float() * (float(hold_reward) + float(progress_boost) * handle_prog)

    event_pen = float(release_event_penalty) * release_event.float()
    lost_pen = float(lost_penalty) * lost.float()

    # if released and door still moves in this short window, penalize a bit
    auto_open = phase & (~keep_ok) & (door_open > float(push_enter_open))
    auto_pen = float(auto_open_penalty) * auto_open.float()

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["keep_after_press/keep_ok_ratio"] = keep_ok.float().mean().detach()
    env.extras["log"]["keep_after_press/contact_ok_ratio"] = contact_ok.float().mean().detach()
    env.extras["log"]["keep_after_press/close_ok_ratio"] = close_ok.float().mean().detach()
    env.extras["log"]["keep_after_press/near_ok_ratio"] = near_ok.float().mean().detach()
    env.extras["log"]["keep_after_press/dist_mean"] = dist.mean().detach()
    env.extras["log"]["keep_after_press/phase_ratio"] = phase.float().mean().detach()

    return hold_rew - event_pen - lost_pen - auto_pen



def unlock_handle_progress_mixed(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    # --- gate ---
    contact_threshold: float = 0.5,
    require_any_contact: bool = True,
    open_width: float = 0.09,
    min_closedness: float = 0.20,
    require_grasp_success: bool = False,
    # --- progress definition ---
    handle_start_pos: float = 0.0,
    reward_stop_pos: float = -0.30,
    # --- delta branch: still encourage "go deeper" ---
    delta_power: float = 1.2,
    ema_alpha: float = 0.7,
    deadzone: float = 5e-5,
    backtrack_penalty: float = 0.01,
    delta_gain: float = 1.2,
    # --- absolute branch: deep is always worth more ---
    abs_power: float = 1.8,
    abs_gain: float = 0.35,
    # --- hold branch: once deep enough, tiny-motion deep press should still score ---
    hold_start_ratio: float = 0.35,   # from here on, give deep-hold reward
    hold_power: float = 1.6,
    hold_gain: float = 0.25,
    # --- final clip ---
    clip: float = 2.0,
) -> torch.Tensor:
    """
    Mixed unlock reward:
      reward = gate * (
          delta_gain * delta_progress_reward
        + abs_gain   * absolute_depth_reward
        + hold_gain  * deep_hold_reward
      )

    - delta_progress_reward:
        rewards newly achieved deeper handle progress, using EMA + backtrack penalty.
    - absolute_depth_reward:
        rewards current depth itself; deeper is worth more.
    - deep_hold_reward:
        once the handle is already in a sufficiently deep region, keep a stable positive reward
        even if the instantaneous motion becomes very small.
    """

    # ------------------------------------------------------------
    # 1) read handle joint pos
    # ------------------------------------------------------------
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = (
        door.data.joint_pos[:, jids[0]]
        if len(jids) == 1
        else door.data.joint_pos[:, jids].mean(dim=-1)
    )

    # ------------------------------------------------------------
    # 2) normalize progress to [0, 1]
    # ------------------------------------------------------------
    denom = float(reward_stop_pos - handle_start_pos)
    if abs(denom) < 1e-9:
        prog = torch.zeros_like(handle_pos)
    else:
        prog = (handle_pos - float(handle_start_pos)) / denom
        prog = torch.clamp(prog, 0.0, 1.0)

    # ------------------------------------------------------------
    # 3) gate: filtered contact + closedness (+ optional grasp_success latch)
    # ------------------------------------------------------------
    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (
        (torch.maximum(fL, fR) > contact_threshold)
        if require_any_contact
        else (torch.minimum(fL, fR) > contact_threshold)
    )

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
    close_ok = closedness > min_closedness

    if require_grasp_success:
        if hasattr(env, "_grasp_success_given"):
            grasp_ok = env._grasp_success_given
        else:
            grasp_ok = torch.zeros_like(contact_ok, dtype=torch.bool)
    else:
        grasp_ok = torch.ones_like(contact_ok, dtype=torch.bool)

    gate = contact_ok & close_ok & grasp_ok

    # ------------------------------------------------------------
    # 4) robust reset detection
    # ------------------------------------------------------------
    N = handle_pos.shape[0]
    device = handle_pos.device

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_unlock_mixed")) or (env._prev_ep_len_unlock_mixed.shape[0] != N):
            env._prev_ep_len_unlock_mixed = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_unlock_mixed) | (env.episode_length_buf == 0)
        env._prev_ep_len_unlock_mixed = env.episode_length_buf.clone()

    # ------------------------------------------------------------
    # 5) EMA buffer for delta branch
    # ------------------------------------------------------------
    if (not hasattr(env, "_unlock_prog_ema")) or (env._unlock_prog_ema.shape[0] != N):
        env._unlock_prog_ema = torch.zeros(N, device=device, dtype=prog.dtype)

    if reset_mask is not None:
        env._unlock_prog_ema = torch.where(reset_mask, prog.detach(), env._unlock_prog_ema)

    prev = env._unlock_prog_ema
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * prog
    env._unlock_prog_ema = new

    # ------------------------------------------------------------
    # 6) delta branch
    # ------------------------------------------------------------
    shaped_prev = torch.clamp(prev, 0.0, 1.0) ** float(delta_power)
    shaped_new = torch.clamp(new, 0.0, 1.0) ** float(delta_power)

    pos = torch.clamp(shaped_new - shaped_prev - float(deadzone), min=0.0)
    neg = torch.clamp(shaped_prev - shaped_new - float(deadzone), min=0.0)
    delta_rew = pos - float(backtrack_penalty) * neg

    # ------------------------------------------------------------
    # 7) absolute branch
    # ------------------------------------------------------------
    abs_rew = torch.clamp(prog, 0.0, 1.0) ** float(abs_power)

    # ------------------------------------------------------------
    # 8) deep-hold branch
    #    once progress is already reasonably deep, keep a stable positive reward
    # ------------------------------------------------------------
    hold_ratio = torch.clamp((prog - float(hold_start_ratio)) / float(max(1e-6, 1.0 - hold_start_ratio)), 0.0, 1.0)
    hold_rew = hold_ratio ** float(hold_power)

    # ------------------------------------------------------------
    # 9) final mixed reward
    # ------------------------------------------------------------
    rew = (
        float(delta_gain) * delta_rew
        + float(abs_gain) * abs_rew
        + float(hold_gain) * hold_rew
    )

    rew = gate.float() * torch.clamp(rew, -float(clip), float(clip))
    return rew



def unlock_success_bonus(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    bonus: float = 30.0,
    handle_threshold: float = -0.3,
    less_than: bool = True,
    hold_steps: int = 3,
    # ---- light gate (match unlock_progress_delta style) ----
    gripper_cfg: SceneEntityCfg | None = None,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.5,
    require_any_contact: bool = True,
    open_width: float = 0.08,
    min_closedness: float = 0.30,
    require_gate: bool = True,
    require_grasp_success: bool = True,
) -> torch.Tensor:
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    unlocked = (handle_pos < handle_threshold) if less_than else (handle_pos > handle_threshold)

    # ---- light gate: filtered contact + closedness ----
    if require_gate and (gripper_cfg is not None):
        fL = filtered_contact_force_norm(env.scene[left_sensor_name])
        fR = filtered_contact_force_norm(env.scene[right_sensor_name])
        contact_ok = (torch.maximum(fL, fR) > contact_threshold) if require_any_contact else (torch.minimum(fL, fR) > contact_threshold)

        _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
        close_ok = closedness > min_closedness
        gate_ok = contact_ok & close_ok
    else:
        gate_ok = torch.ones_like(unlocked, dtype=torch.bool)

    # Require episode-level grasp_success latch first to avoid accidental unlock triggers.
    if require_grasp_success:
        if hasattr(env, "_grasp_success_given"):
            grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
        else:
            grasp_ok = torch.zeros_like(unlocked, dtype=torch.bool)
    else:
        grasp_ok = torch.ones_like(unlocked, dtype=torch.bool)

    # sustained + one-shot: require `hold_steps` consecutive steps of
    # (grasp_success_latched & unlocked & existing_gate)
    good = grasp_ok & unlocked & gate_ok

    N = handle_pos.shape[0]
    device = handle_pos.device
    if (not hasattr(env, "_unlock_success_counter")) or (env._unlock_success_counter.shape[0] != N):
        env._unlock_success_counter = torch.zeros(N, device=device, dtype=torch.int32)
    if (not hasattr(env, "_unlock_success_given")) or (env._unlock_success_given.shape[0] != N):
        env._unlock_success_given = torch.zeros(N, device=device, dtype=torch.bool)

    # ---- robust reset detection ----
    reset_mask = None

    # 1) 优先用 reset_buf
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    # 2) fallback: 用 episode_length 的“回绕”检测 reset
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_unlock_success")) or (env._prev_ep_len_unlock_success.shape[0] != N):
            env._prev_ep_len_unlock_success = env.episode_length_buf.clone()

        reset_mask = (env.episode_length_buf < env._prev_ep_len_unlock_success) | (env.episode_length_buf == 0)
        env._prev_ep_len_unlock_success = env.episode_length_buf.clone()

    if reset_mask is not None:
        env._unlock_success_counter = torch.where(
            reset_mask, torch.zeros_like(env._unlock_success_counter), env._unlock_success_counter
        )
        env._unlock_success_given = torch.where(
            reset_mask, torch.zeros_like(env._unlock_success_given), env._unlock_success_given
        )

    env._unlock_success_counter = torch.where(
        good, env._unlock_success_counter + 1, torch.zeros_like(env._unlock_success_counter)
    )
    success_now = env._unlock_success_counter >= int(max(1, hold_steps))

    give = success_now & (~env._unlock_success_given)
    env._unlock_success_given = env._unlock_success_given | give

    # ---- archive push for staged reset (unlock) ----
    idx = torch.nonzero(give).squeeze(-1)
    if idx.numel() > 0:
        from .stage import push_archive_from_env
        push_archive_from_env(
            env=env,
            name="_archive_unlock",
            env_ids=idx,
            cap=512,
            robot_cfg=SceneEntityCfg("robot"),
            door_cfg=SceneEntityCfg("door"),
            store_unlock_flag=True,
        )

    return give.float() * float(bonus)


def push_door_progress_after_unlock(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    # --- unlock gate ---
    handle_joint_cfg: SceneEntityCfg | None = None,
    handle_threshold: float = -0.3,
    handle_less_than: bool = True,
    use_unlock_success_latch: bool = True,
    # --- door progress definition ---
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_target: float = 0.35,
    # --- reward scales ---
    delta_scale: float = 1.0,
    abs_scale: float = 0.2,
    # --- anti-jitter ---
    ema_alpha: float = 0.25,
    deadzone: float = 1e-4,
    backtrack_penalty: float = 0.2,
    clip: float = 1.0,
) -> torch.Tensor:
    # ------------------------------------------------------------------
    # 1) read door joint position
    # ------------------------------------------------------------------
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    # ------------------------------------------------------------------
    # 2) unlock phase gate only
    # ------------------------------------------------------------------
    if use_unlock_success_latch and hasattr(env, "_unlock_success_given"):
        unlocked_phase = env._unlock_success_given
    else:
        if handle_joint_cfg is None:
            unlocked_phase = torch.ones_like(door_open, dtype=torch.bool)
        else:
            handle_asset: Articulation = env.scene[handle_joint_cfg.name]
            hjids = handle_joint_cfg.joint_ids
            handle_pos = handle_asset.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else handle_asset.data.joint_pos[:, hjids].mean(dim=-1)
            unlocked_phase = (handle_pos < handle_threshold) if handle_less_than else (handle_pos > handle_threshold)

    # ------------------------------------------------------------------
    # 3) reset detection
    # ------------------------------------------------------------------
    N = door_open.shape[0]
    device = door_open.device
    dtype = door_open.dtype

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_open_door_free")) or (env._prev_ep_len_open_door_free.shape[0] != N):
            env._prev_ep_len_open_door_free = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_open_door_free) | (env.episode_length_buf == 0)
        env._prev_ep_len_open_door_free = env.episode_length_buf.clone()

    # ------------------------------------------------------------------
    # 4) EMA buffer
    # ------------------------------------------------------------------
    if (not hasattr(env, "_door_open_ema_free")) or (env._door_open_ema_free.shape[0] != N):
        env._door_open_ema_free = torch.zeros(N, device=device, dtype=dtype)

    if reset_mask is not None:
        env._door_open_ema_free = torch.where(reset_mask, door_open.detach(), env._door_open_ema_free)

    prev = env._door_open_ema_free
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * door_open
    env._door_open_ema_free = new

    d = new - prev
    pos = torch.clamp(d - float(deadzone), min=0.0)
    neg = torch.clamp(-d - float(deadzone), min=0.0)

    delta_rew = pos - float(backtrack_penalty) * neg
    delta_rew = torch.clamp(delta_rew, -float(clip), float(clip))

    abs_open = torch.clamp(new / float(max(1e-6, door_open_target)), 0.0, 1.0)

    rew = float(delta_scale) * delta_rew + float(abs_scale) * abs_open
    return unlocked_phase.float() * rew


def near_unlock_stall_penalty(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg,
    door_joint_cfg: SceneEntityCfg,
    enter_depth: float = 0.25,
    exit_depth: float = 0.22,
    grace_steps: int = 60,
    ramp_steps: int = 60,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_progress_threshold: float = 0.01,
    max_penalty: float = 1.0,
    less_than: bool = True,
) -> torch.Tensor:
    door: Articulation = env.scene[handle_joint_cfg.name]
    hjids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, hjids[0]] if len(hjids) == 1 else door.data.joint_pos[:, hjids].mean(dim=-1)

    djids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, djids[0]] if len(djids) == 1 else door.data.joint_pos[:, djids].mean(dim=-1)
    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    N = handle_pos.shape[0]
    device = handle_pos.device

    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(N, dtype=torch.bool, device=device)
    physical_unlocked = _physical_door_unlocked(env)

    press_depth = -handle_pos if less_than else handle_pos
    enter = press_depth >= float(enter_depth)
    remain = press_depth >= float(exit_depth)

    if (not hasattr(env, "_near_unlock_stall_active")) or (env._near_unlock_stall_active.shape[0] != N):
        env._near_unlock_stall_active = torch.zeros(N, dtype=torch.bool, device=device)
    if (not hasattr(env, "_near_unlock_stall_counter")) or (env._near_unlock_stall_counter.shape[0] != N):
        env._near_unlock_stall_counter = torch.zeros(N, dtype=torch.int32, device=device)

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_near_unlock_stall")) or (
            env._prev_ep_len_near_unlock_stall.shape[0] != N
        ):
            env._prev_ep_len_near_unlock_stall = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_near_unlock_stall)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_near_unlock_stall = env.episode_length_buf.clone()

    active = grasp_ok & (~physical_unlocked) & (enter | (env._near_unlock_stall_active & remain))

    if reset_mask is not None:
        active = active & (~reset_mask)
        env._near_unlock_stall_active = torch.where(
            reset_mask, torch.zeros_like(env._near_unlock_stall_active), active
        )
        env._near_unlock_stall_counter = torch.where(
            reset_mask, torch.zeros_like(env._near_unlock_stall_counter), env._near_unlock_stall_counter
        )
    else:
        env._near_unlock_stall_active = active

    clear_counter = physical_unlocked | (~active)
    env._near_unlock_stall_counter = torch.where(
        clear_counter,
        torch.zeros_like(env._near_unlock_stall_counter),
        env._near_unlock_stall_counter + 1,
    )

    door_not_opening = door_open < float(door_progress_threshold)
    over_steps = torch.clamp(env._near_unlock_stall_counter.float() - float(grace_steps), min=0.0)
    stall_scale = torch.tanh(over_steps / float(max(int(ramp_steps), 1)))
    penalty = -float(max_penalty) * stall_scale * active.float() * door_not_opening.float()

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["stall_after_press/deep_press_active_ratio"] = active.float().mean().detach()
    env.extras["log"]["stall_after_press/counter_mean"] = env._near_unlock_stall_counter.float().mean().detach()
    env.extras["log"]["stall_after_press/counter_max"] = env._near_unlock_stall_counter.float().max().detach()
    env.extras["log"]["stall_after_press/penalty_mean"] = penalty.mean().detach()
    env.extras["log"]["stall_after_press/penalty_active_ratio"] = (penalty < 0.0).float().mean().detach()
    env.extras["log"]["handle_debug/handle_pos_mean"] = handle_pos.mean().detach()
    env.extras["log"]["handle_debug/handle_pos_min"] = handle_pos.min().detach()
    env.extras["log"]["handle_debug/near_unlock_ratio_025_030"] = (
        ((handle_pos <= -0.25) & (handle_pos > -0.30)).float().mean().detach()
    )
    env.extras["log"]["handle_debug/cross_unlock_threshold_ratio"] = (handle_pos <= -0.30).float().mean().detach()

    return penalty


def physical_unlock_transition_bonus(
    env: "ManagerBasedRLEnv",
    bonus: float = 10.0,
) -> torch.Tensor:
    physical_unlocked = _physical_door_unlocked(env)
    N = physical_unlocked.shape[0]
    device = physical_unlocked.device

    if (not hasattr(env, "_unlock_success_given")) or (env._unlock_success_given.shape[0] != N):
        env._unlock_success_given = torch.zeros(N, dtype=torch.bool, device=device)

    newly_unlocked = physical_unlocked & (~env._unlock_success_given.to(dtype=torch.bool))
    reward = newly_unlocked.float() * float(bonus)
    env._unlock_success_given = env._unlock_success_given | newly_unlocked

    idx = torch.nonzero(newly_unlocked).squeeze(-1)
    if idx.numel() > 0:
        from .stage import push_archive_from_env
        push_archive_from_env(
            env=env,
            name="_archive_unlock",
            env_ids=idx,
            cap=512,
            robot_cfg=SceneEntityCfg("robot"),
            door_cfg=SceneEntityCfg("door"),
            store_unlock_flag=True,
        )

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["transition/physical_unlocked_ratio"] = physical_unlocked.float().mean().detach()
    env.extras["log"]["transition/newly_unlocked_ratio"] = newly_unlocked.float().mean().detach()
    env.extras["log"]["transition/unlock_bonus_mean"] = reward.mean().detach()

    return reward


def release_after_unlock_failure(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
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
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_threshold: float = 0.35,
    fail_steps: int = 30,
    after_unlock_grace_steps: int = 10,
) -> torch.Tensor:
    N = env.num_envs
    device = env.device

    door: Articulation = env.scene[door_joint_cfg.name]
    djids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, djids[0]] if len(djids) == 1 else door.data.joint_pos[:, djids].mean(dim=-1)
    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    fL = filtered_contact_force_norm(env.scene[left_sensor_name])
    fR = filtered_contact_force_norm(env.scene[right_sensor_name])
    contact_ok = (
        (torch.maximum(fL, fR) > float(contact_threshold))
        if require_any_contact
        else (torch.minimum(fL, fR) > float(contact_threshold))
    )

    _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
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

    if hasattr(env, "_unlock_success_given"):
        unlocked = env._unlock_success_given.to(dtype=torch.bool)
    else:
        unlocked = _physical_door_unlocked(env)
    door_not_open = door_open < float(door_open_threshold)

    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_release_after_unlock")) or (
            env._prev_ep_len_release_after_unlock.shape[0] != N
        ):
            env._prev_ep_len_release_after_unlock = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_release_after_unlock)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_release_after_unlock = env.episode_length_buf.clone()

    if (not hasattr(env, "_release_after_unlock_counter")) or (
        env._release_after_unlock_counter.shape[0] != N
    ):
        env._release_after_unlock_counter = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_release_after_unlock_grace")) or (
        env._release_after_unlock_grace.shape[0] != N
    ):
        env._release_after_unlock_grace = torch.zeros(N, dtype=torch.int32, device=device)
    if (not hasattr(env, "_prev_release_after_unlock_unlocked")) or (
        env._prev_release_after_unlock_unlocked.shape[0] != N
    ):
        env._prev_release_after_unlock_unlocked = torch.zeros(N, dtype=torch.bool, device=device)

    newly_unlocked = unlocked & (~env._prev_release_after_unlock_unlocked)
    env._prev_release_after_unlock_unlocked = unlocked

    env._release_after_unlock_grace = torch.where(
        newly_unlocked,
        torch.full_like(env._release_after_unlock_grace, int(after_unlock_grace_steps)),
        torch.clamp(env._release_after_unlock_grace - 1, min=0),
    )

    if reset_mask is not None:
        env._release_after_unlock_counter = torch.where(
            reset_mask, torch.zeros_like(env._release_after_unlock_counter), env._release_after_unlock_counter
        )
        env._release_after_unlock_grace = torch.where(
            reset_mask, torch.zeros_like(env._release_after_unlock_grace), env._release_after_unlock_grace
        )
        env._prev_release_after_unlock_unlocked = torch.where(
            reset_mask, torch.zeros_like(env._prev_release_after_unlock_unlocked), env._prev_release_after_unlock_unlocked
        )

    bad = unlocked & door_not_open & (env._release_after_unlock_grace == 0) & (~keep_ok)
    env._release_after_unlock_counter = torch.where(
        bad,
        env._release_after_unlock_counter + 1,
        torch.zeros_like(env._release_after_unlock_counter),
    )
    done = env._release_after_unlock_counter >= int(max(1, fail_steps))

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["termination/release_after_unlock_bad_ratio"] = bad.float().mean().detach()
    env.extras["log"]["termination/release_after_unlock_counter_mean"] = (
        env._release_after_unlock_counter.float().mean().detach()
    )
    env.extras["log"]["termination/release_after_unlock_done_ratio"] = done.float().mean().detach()
    env.extras["log"]["termination/release_after_unlock_keep_ok_ratio"] = keep_ok.float().mean().detach()
    env.extras["log"]["termination/release_after_unlock_grace_mean"] = (
        env._release_after_unlock_grace.float().mean().detach()
    )

    return done


def push_door_progress_after_unlock_success_only(
    env: "ManagerBasedRLEnv",
    door_joint_cfg: SceneEntityCfg,
    # 只在 unlock_success 之后生效
    require_unlock_success_latch: bool = True,
    # gate: filtered contact + closedness, same style as unlock process
    gripper_cfg: SceneEntityCfg | None = None,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
    contact_threshold: float = 0.5,
    require_any_contact: bool = True,
    open_width: float = 0.09,
    min_closedness: float = 0.50,
    require_gate: bool = True,
    # 门开度定义
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    door_open_target: float = 0.35,
    # 奖励
    delta_scale: float = 1.0,
    abs_scale: float = 0.2,
    # 平滑/防抖
    ema_alpha: float = 0.25,
    deadzone: float = 1e-4,
    backtrack_penalty: float = 0.2,
    clip: float = 1.0,
) -> torch.Tensor:
    door: Articulation = env.scene[door_joint_cfg.name]
    jids = door_joint_cfg.joint_ids
    door_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)

    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    N = door_open.shape[0]
    device = door_open.device
    dtype = door_open.dtype

    # -----------------------------
    # gate: ONLY after unlock_success
    # -----------------------------
    if require_unlock_success_latch:
        if hasattr(env, "_unlock_success_given"):
            gate = env._unlock_success_given
        else:
            gate = torch.zeros(N, dtype=torch.bool, device=device)
    else:
        gate = torch.ones(N, dtype=torch.bool, device=device)

    if require_gate and (gripper_cfg is not None):
        fL = filtered_contact_force_norm(env.scene[left_sensor_name])
        fR = filtered_contact_force_norm(env.scene[right_sensor_name])
        contact_ok = (
            (torch.maximum(fL, fR) > contact_threshold)
            if require_any_contact
            else (torch.minimum(fL, fR) > contact_threshold)
        )
        _, closedness = _gripper_width_and_closedness(env, gripper_cfg, open_width=open_width)
        close_ok = closedness > min_closedness
        gate = gate & contact_ok & close_ok

    # -----------------------------
    # robust reset detection
    # -----------------------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None

    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_open_after_unlock_success")) or (
            env._prev_ep_len_open_after_unlock_success.shape[0] != N
        ):
            env._prev_ep_len_open_after_unlock_success = env.episode_length_buf.clone()
        reset_mask = (
            (env.episode_length_buf < env._prev_ep_len_open_after_unlock_success)
            | (env.episode_length_buf == 0)
        )
        env._prev_ep_len_open_after_unlock_success = env.episode_length_buf.clone()

    # -----------------------------
    # EMA buffer
    # -----------------------------
    if (not hasattr(env, "_door_open_ema_after_unlock_success")) or (
        env._door_open_ema_after_unlock_success.shape[0] != N
    ):
        env._door_open_ema_after_unlock_success = torch.zeros(N, device=device, dtype=dtype)

    if reset_mask is not None:
        env._door_open_ema_after_unlock_success = torch.where(
            reset_mask,
            door_open.detach(),
            env._door_open_ema_after_unlock_success,
        )

    prev = env._door_open_ema_after_unlock_success
    new = (1.0 - float(ema_alpha)) * prev + float(ema_alpha) * door_open
    env._door_open_ema_after_unlock_success = new

    d = new - prev
    pos = torch.clamp(d - float(deadzone), min=0.0)
    neg = torch.clamp(-d - float(deadzone), min=0.0)

    delta_rew = pos - float(backtrack_penalty) * neg
    delta_rew = torch.clamp(delta_rew, -float(clip), float(clip))

    abs_open = torch.clamp(new / float(max(1e-6, door_open_target)), 0.0, 1.0)

    rew = float(delta_scale) * delta_rew + float(abs_scale) * abs_open
    return gate.float() * rew


def _weighted_reward(
    func,
    env: "ManagerBasedRLEnv",
    weight: float,
    params: dict | None,
) -> torch.Tensor:
    if params is None:
        params = {}
    params = _resolve_nested_scene_entity_cfgs(env, dict(params))
    return float(weight) * func(env, **params)


def _scene_entity_cfg_needs_resolve(cfg: object) -> bool:
    for attr_name in ("body_ids", "joint_ids", "fixed_tendon_ids", "object_collection_ids"):
        if hasattr(cfg, attr_name) and isinstance(getattr(cfg, attr_name), slice):
            return True
    return False


def _resolve_nested_scene_entity_cfgs(env: "ManagerBasedRLEnv", value):
    if isinstance(value, dict):
        return {key: _resolve_nested_scene_entity_cfgs(env, item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_nested_scene_entity_cfgs(env, item) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_nested_scene_entity_cfgs(env, item) for item in value)
    if hasattr(value, "resolve") and _scene_entity_cfg_needs_resolve(value):
        value.resolve(env.scene)
    return value


def _log_stage_reward_scalar(env: "ManagerBasedRLEnv", name: str, value: torch.Tensor) -> None:
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"][f"stage_gated_reward/{name}"] = value.mean().detach()


def _read_handle_unlock_flag(
    env: "ManagerBasedRLEnv",
    handle_joint_cfg: SceneEntityCfg | None,
    handle_threshold: float,
    less_than: bool,
) -> torch.Tensor:
    if handle_joint_cfg is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    handle_joint_cfg = _resolve_nested_scene_entity_cfgs(env, handle_joint_cfg)
    door: Articulation = env.scene[handle_joint_cfg.name]
    jids = handle_joint_cfg.joint_ids
    handle_pos = door.data.joint_pos[:, jids[0]] if len(jids) == 1 else door.data.joint_pos[:, jids].mean(dim=-1)
    return (handle_pos < float(handle_threshold)) if less_than else (handle_pos > float(handle_threshold))


def stage_gated_door_reward(
    env: "ManagerBasedRLEnv",
    enable_stage_gated_reward: bool = True,
    stage0_only_reward: bool = False,
    pre_grasp_cap: float = 1.0,
    align_grasp_weight: float = 0.0,
    align_grasp_params: dict | None = None,
    approach_handle_weight: float = 0.0,
    approach_handle_params: dict | None = None,
    close_when_ready_weight: float = 0.0,
    close_when_ready_params: dict | None = None,
    grasp_handle_weight: float = 0.0,
    grasp_handle_params: dict | None = None,
    grasp_success_weight: float = 0.0,
    grasp_success_params: dict | None = None,
    press_handle_weight: float = 0.0,
    press_handle_params: dict | None = None,
    keep_handle_after_press_weight: float = 0.0,
    keep_handle_after_press_params: dict | None = None,
    grasp_quality_keep_weight: float = 2.0,
    grasp_quality_keep_params: dict | None = None,
    grasp_quality_gate_floor: float = 0.15,
    stall_after_grasp_weight: float = 0.0,
    stall_after_grasp_params: dict | None = None,
    stall_after_press_weight: float = 0.0,
    stall_after_press_params: dict | None = None,
    unlock_progress_weight: float = 0.0,
    unlock_progress_params: dict | None = None,
    unlock_transition_weight: float = 0.0,
    unlock_transition_params: dict | None = None,
    push_door_weight: float = 0.0,
    push_door_params: dict | None = None,
) -> torch.Tensor:
    """Aggregate DoorBot rewards with explicit stage masks."""

    align_reward = _weighted_reward(align_grasp_pose_v2, env, align_grasp_weight, align_grasp_params)
    approach_reward = _weighted_reward(approach_handle_inv_square, env, approach_handle_weight, approach_handle_params)
    close_when_ready_reward = _weighted_reward(
        close_gripper_shaping_when_ready, env, close_when_ready_weight, close_when_ready_params
    )
    grasp_handle_reward_term = _weighted_reward(
        grasp_handle_reward_preunlock_only, env, grasp_handle_weight, grasp_handle_params
    )

    stage0_reward = (
        align_reward
        + approach_reward
        + close_when_ready_reward
        + grasp_handle_reward_term
    )

    if stage0_only_reward:
        zero_reward = torch.zeros_like(stage0_reward)
        one_mask = torch.ones_like(stage0_reward)
        zero_mask = torch.zeros_like(stage0_reward)

        _log_stage_reward_scalar(env, "align", align_reward)
        _log_stage_reward_scalar(env, "approach", approach_reward)
        _log_stage_reward_scalar(env, "close_when_ready", close_when_ready_reward)
        _log_stage_reward_scalar(env, "grasp_handle", grasp_handle_reward_term)
        _log_stage_reward_scalar(env, "press_handle", zero_reward)
        _log_stage_reward_scalar(env, "keep_handle_after_press", zero_reward)
        _log_stage_reward_scalar(env, "stall_after_grasp", zero_reward)
        _log_stage_reward_scalar(env, "stall_after_press", zero_reward)
        _log_stage_reward_scalar(env, "unlock_progress", zero_reward)
        _log_stage_reward_scalar(env, "unlock_transition", zero_reward)
        _log_stage_reward_scalar(env, "push_door", zero_reward)
        _log_stage_reward_scalar(env, "stage0_reward", stage0_reward)
        _log_stage_reward_scalar(env, "stage1_reward", zero_reward)
        _log_stage_reward_scalar(env, "stage2_reward", zero_reward)
        _log_stage_reward_scalar(env, "stage0_mask_ratio", one_mask)
        _log_stage_reward_scalar(env, "stage1_mask_ratio", zero_mask)
        _log_stage_reward_scalar(env, "stage2_mask_ratio", zero_mask)

        return stage0_reward

    if grasp_success_params is None:
        grasp_success_params = {}
    grasp_success_call_params = _resolve_nested_scene_entity_cfgs(env, dict(grasp_success_params))
    if enable_stage_gated_reward:
        grasp_success_call_params["bonus"] = 0.0
    grasp_success_reward = float(grasp_success_weight) * grasp_success_bonus(env, **grasp_success_call_params)

    if hasattr(env, "_grasp_success_given"):
        grasp_ok = env._grasp_success_given.to(dtype=torch.bool)
    else:
        grasp_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    unlock_transition_reward = _weighted_reward(
        physical_unlock_transition_bonus, env, unlock_transition_weight, unlock_transition_params
    )
    unlock_flag = _physical_door_unlocked(env)

    press_handle_reward = _weighted_reward(
        press_handle_after_grasp_vel, env, press_handle_weight, press_handle_params
    )
    keep_handle_after_press_reward = _weighted_reward(
        anti_release_after_press_to_open, env, keep_handle_after_press_weight, keep_handle_after_press_params
    )
    if grasp_quality_keep_params is None:
        zero = torch.zeros(env.num_envs, dtype=press_handle_reward.dtype, device=env.device)
        false = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        quality_terms = {
            "active": false,
            "quality": zero,
            "near_score": zero,
            "wrap_score": zero,
            "open_align": zero,
            "approach_dot_raw": zero,
            "approach_align": zero,
            "pose_score": zero,
            "contact_score": zero,
            "contact_ok": false,
            "balance_score": zero,
            "closure_score": zero,
            "closedness": zero,
            "tcp_dist": zero,
            "f_left": zero,
            "f_right": zero,
            "f_min": zero,
            "f_max": zero,
            "closed_no_contact": false,
            "single_finger": false,
        }
        closed_no_contact_penalty = 0.5
        single_finger_penalty = 0.5
    else:
        quality_compute_params = dict(grasp_quality_keep_params)
        closed_no_contact_penalty = float(quality_compute_params.pop("closed_no_contact_penalty", 0.5))
        single_finger_penalty = float(quality_compute_params.pop("single_finger_penalty", 0.5))
        quality_compute_params = _resolve_nested_scene_entity_cfgs(env, quality_compute_params)
        quality_terms = compute_stage1_grasp_quality(env, **quality_compute_params)
    grasp_quality = quality_terms["quality"]
    quality_gate = (
        float(grasp_quality_gate_floor)
        + (1.0 - float(grasp_quality_gate_floor)) * grasp_quality
    ).clamp(0.0, 1.0)
    grasp_quality_keep_reward_term = float(grasp_quality_keep_weight) * grasp_quality
    quality_penalty = (
        -closed_no_contact_penalty * quality_terms["closed_no_contact"].float()
        -single_finger_penalty * quality_terms["single_finger"].float()
    )
    gated_press_handle_reward = press_handle_reward * quality_gate

    stall_after_grasp_reward = _weighted_reward(
        stall_penalty_after_grasp_pos, env, stall_after_grasp_weight, stall_after_grasp_params
    )
    stall_after_press_reward = _weighted_reward(
        near_unlock_stall_penalty, env, stall_after_press_weight, stall_after_press_params
    )
    unlock_progress_reward = _weighted_reward(
        unlock_handle_progress_mixed, env, unlock_progress_weight, unlock_progress_params
    )
    gated_unlock_progress_reward = unlock_progress_reward * quality_gate

    push_door_reward = _weighted_reward(
        push_door_progress_after_unlock_success_only, env, push_door_weight, push_door_params
    )

    stage1_reward = (
        (float(pre_grasp_cap) * grasp_quality)
        + grasp_quality_keep_reward_term
        + keep_handle_after_press_reward
        + gated_press_handle_reward
        + stall_after_grasp_reward
        + stall_after_press_reward
        + gated_unlock_progress_reward
        + quality_penalty
    )
    stage2_reward = push_door_reward + 0.5 * keep_handle_after_press_reward

    stage0_mask = ~grasp_ok
    stage1_mask = grasp_ok & (~unlock_flag)
    stage2_mask = unlock_flag

    if enable_stage_gated_reward:
        masked_stage_reward = (
            stage0_mask.float() * stage0_reward
            + stage1_mask.float() * stage1_reward
            + stage2_mask.float() * stage2_reward
        )
        total_reward = masked_stage_reward + unlock_transition_reward
    else:
        total_reward = (
            stage0_reward
            + grasp_success_reward
            + press_handle_reward
            + keep_handle_after_press_reward
            + stall_after_grasp_reward
            + stall_after_press_reward
            + unlock_progress_reward
            + unlock_transition_reward
            + push_door_reward
        )

    _log_stage_reward_scalar(env, "align", align_reward)
    _log_stage_reward_scalar(env, "approach", approach_reward)
    _log_stage_reward_scalar(env, "close_when_ready", close_when_ready_reward)
    _log_stage_reward_scalar(env, "grasp_handle", grasp_handle_reward_term)
    _log_stage_reward_scalar(env, "press_handle", press_handle_reward)
    _log_stage_reward_scalar(env, "grasp_quality_keep", grasp_quality_keep_reward_term)
    _log_stage_reward_scalar(env, "grasp_quality_penalty", quality_penalty)
    _log_stage_reward_scalar(env, "press_handle_raw", press_handle_reward)
    _log_stage_reward_scalar(env, "press_handle_gated", gated_press_handle_reward)
    _log_stage_reward_scalar(env, "keep_handle_after_press", keep_handle_after_press_reward)
    _log_stage_reward_scalar(env, "stall_after_grasp", stall_after_grasp_reward)
    _log_stage_reward_scalar(env, "stall_after_press", stall_after_press_reward)
    _log_stage_reward_scalar(env, "unlock_progress", unlock_progress_reward)
    _log_stage_reward_scalar(env, "unlock_progress_raw", unlock_progress_reward)
    _log_stage_reward_scalar(env, "unlock_progress_gated", gated_unlock_progress_reward)
    _log_stage_reward_scalar(env, "unlock_transition", unlock_transition_reward)
    _log_stage_reward_scalar(env, "push_door", push_door_reward)
    _log_stage_reward_scalar(env, "stage0_reward", stage0_reward)
    _log_stage_reward_scalar(env, "stage1_reward", stage1_reward)
    _log_stage_reward_scalar(env, "stage2_reward", stage2_reward)
    _log_stage_reward_scalar(env, "stage0_mask_ratio", stage0_mask.float())
    _log_stage_reward_scalar(env, "stage1_mask_ratio", stage1_mask.float())
    _log_stage_reward_scalar(env, "stage2_mask_ratio", stage2_mask.float())

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    active = quality_terms["active"]
    env.extras["log"]["grasp_quality/active_ratio"] = active.float().mean().detach()
    env.extras["log"]["grasp_quality/quality_mean"] = _active_mean(quality_terms["quality"], active)
    env.extras["log"]["grasp_quality/near_score_mean"] = _active_mean(quality_terms["near_score"], active)
    env.extras["log"]["grasp_quality/tcp_dist_mean"] = _active_mean(quality_terms["tcp_dist"], active)
    env.extras["log"]["grasp_quality/tcp_dist_max"] = _active_max(quality_terms["tcp_dist"], active)
    env.extras["log"]["grasp_quality/wrap_score_mean"] = _active_mean(quality_terms["wrap_score"], active)
    env.extras["log"]["grasp_quality/open_align_mean"] = _active_mean(quality_terms["open_align"], active)
    env.extras["log"]["grasp_quality/approach_dot_raw_mean"] = _active_mean(quality_terms["approach_dot_raw"], active)
    env.extras["log"]["grasp_quality/approach_align_mean"] = _active_mean(quality_terms["approach_align"], active)
    env.extras["log"]["grasp_quality/pose_score_mean"] = _active_mean(quality_terms["pose_score"], active)
    env.extras["log"]["grasp_quality/contact_score_mean"] = _active_mean(quality_terms["contact_score"], active)
    env.extras["log"]["grasp_quality/contact_ok_ratio"] = _active_mean(quality_terms["contact_ok"].float(), active)
    env.extras["log"]["grasp_quality/balance_score_mean"] = _active_mean(quality_terms["balance_score"], active)
    env.extras["log"]["grasp_quality/closure_score_mean"] = _active_mean(quality_terms["closure_score"], active)
    env.extras["log"]["grasp_quality/closedness_mean"] = _active_mean(quality_terms["closedness"], active)
    env.extras["log"]["grasp_quality/quality_gate_mean"] = _active_mean(quality_gate, active)
    env.extras["log"]["grasp_quality/closed_no_contact_ratio"] = _active_mean(
        quality_terms["closed_no_contact"].float(), active
    )
    env.extras["log"]["grasp_quality/single_finger_ratio"] = _active_mean(quality_terms["single_finger"].float(), active)
    env.extras["log"]["grasp_quality/f_left_mean"] = _active_mean(quality_terms["f_left"], active)
    env.extras["log"]["grasp_quality/f_right_mean"] = _active_mean(quality_terms["f_right"], active)
    env.extras["log"]["grasp_quality/f_min_mean"] = _active_mean(quality_terms["f_min"], active)
    env.extras["log"]["grasp_quality/f_max_mean"] = _active_mean(quality_terms["f_max"], active)

    return total_reward
