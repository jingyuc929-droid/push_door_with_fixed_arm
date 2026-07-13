from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv


# -------------------------
# quaternion helpers (wxyz)
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


def quat_normalize(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (torch.norm(q, dim=-1, keepdim=True) + eps)


# -------------------------
# generic helpers
# -------------------------
def _get_single_body_pose_w(env: ManagerBasedRLEnv, cfg: SceneEntityCfg):
    asset: Articulation = env.scene[cfg.name]
    bid = cfg.body_ids[0]
    pos = asset.data.body_pos_w[:, bid, :]    # [N, 3]
    quat = asset.data.body_quat_w[:, bid, :]  # [N, 4], wxyz
    return pos, quat


def _expand_vec3(x, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(x, dtype=like.dtype, device=like.device).view(1, 3).expand(like.shape[0], 3)


def _expand_quat(x, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor(x, dtype=like.dtype, device=like.device).view(1, 4).expand(like.shape[0], 4)


def _get_single_body_pose_w_with_offset(
    env: ManagerBasedRLEnv,
    cfg: SceneEntityCfg,
    offset_pos=(0.0, 0.0, 0.0),
    offset_quat=(1.0, 0.0, 0.0, 0.0),
):
    """Get pose of a virtual frame rigidly attached to one body.

    World pose:
      p_tcp = p_body + R_body * offset_pos
      q_tcp = q_body * offset_quat
    """
    body_pos, body_quat = _get_single_body_pose_w(env, cfg)

    off_pos = _expand_vec3(offset_pos, body_pos)
    off_quat = _expand_quat(offset_quat, body_quat)

    tcp_pos = body_pos + quat_rotate(body_quat, off_pos)
    tcp_quat = quat_normalize(quat_mul(body_quat, off_quat))
    return tcp_pos, tcp_quat


def _world_vector_to_base(env: ManagerBasedRLEnv, vector_w: torch.Tensor, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return quat_rotate(quat_conjugate(robot.data.root_quat_w), vector_w)


def _handle_target_point_w(
    env: ManagerBasedRLEnv,
    handle_cfg: SceneEntityCfg,
    handle_offset_h=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    handle_pos_w, handle_quat_w = _get_single_body_pose_w(env, handle_cfg)
    offset_h = _expand_vec3(handle_offset_h, handle_pos_w)
    return handle_pos_w + quat_rotate(handle_quat_w, offset_h)


def _doorway_center_w(
    env: ManagerBasedRLEnv,
    door_cfg: SceneEntityCfg,
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_z: float = 0.0,
) -> torch.Tensor:
    door: Articulation = env.scene[door_cfg.name]
    center_d = torch.tensor(
        (doorway_center_xy[0], doorway_center_xy[1], doorway_z),
        dtype=door.data.root_pos_w.dtype,
        device=door.data.root_pos_w.device,
    ).view(1, 3).expand(env.num_envs, 3)
    return door.data.root_pos_w + quat_rotate(door.data.root_quat_w, center_d)


def _doorway_forward_w(
    env: ManagerBasedRLEnv,
    door_cfg: SceneEntityCfg,
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
) -> torch.Tensor:
    door: Articulation = env.scene[door_cfg.name]
    axis_d = torch.tensor(
        (doorway_forward_axis[0], doorway_forward_axis[1], 0.0),
        dtype=door.data.root_pos_w.dtype,
        device=door.data.root_pos_w.device,
    ).view(1, 3).expand(env.num_envs, 3)
    return quat_rotate(door.data.root_quat_w, axis_d)


def _contact_force_norm(sensor: ContactSensor) -> torch.Tensor:
    f = sensor.data.net_forces_w
    if f.ndim == 3:  # [N, B, 3]
        return torch.norm(f, dim=-1).max(dim=-1).values
    elif f.ndim == 2:  # [N, 3]
        return torch.norm(f, dim=-1)
    else:
        raise RuntimeError(f"Unexpected net_forces_w shape: {tuple(f.shape)}")


def _filtered_contact_force_norm_from_sensor(env: ManagerBasedRLEnv, sensor_name: str | None) -> torch.Tensor:
    if not sensor_name:
        return torch.zeros((env.num_envs,), device=env.device)
    try:
        sensor: ContactSensor = env.scene[sensor_name]
    except Exception:
        return torch.zeros((env.num_envs,), device=env.device)

    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is not None:
        if fm.ndim == 4:
            return torch.linalg.norm(fm.sum(dim=2), dim=-1).amax(dim=1)
        if fm.ndim == 3:
            return torch.linalg.norm(fm.sum(dim=1), dim=-1)
        return torch.zeros((env.num_envs,), device=env.device)
    return _contact_force_norm(sensor)


def normalized_contact_force(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    force_ref: float = 50.0,
) -> torch.Tensor:
    """Return clamp(filtered_contact_force_norm / force_ref, 0, 1) as [N, 1]."""
    force = _filtered_contact_force_norm_from_sensor(env, sensor_name)
    return torch.clamp(force / max(float(force_ref), 1.0e-6), 0.0, 1.0).unsqueeze(-1)


def body_door_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_name: str = "body_door_contact",
    force_ref: float = 50.0,
) -> torch.Tensor:
    return normalized_contact_force(env, sensor_name, force_ref)


def leg_door_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_name: str = "leg_door_contact",
    force_ref: float = 50.0,
) -> torch.Tensor:
    return normalized_contact_force(env, sensor_name, force_ref)


def body_door_frame_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_name: str = "body_door_frame_contact",
    force_ref: float = 50.0,
) -> torch.Tensor:
    return normalized_contact_force(env, sensor_name, force_ref)


def leg_door_frame_contact_force_norm(
    env: ManagerBasedRLEnv,
    sensor_name: str = "leg_door_frame_contact",
    force_ref: float = 50.0,
) -> torch.Tensor:
    return normalized_contact_force(env, sensor_name, force_ref)


def _get_action_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    action = getattr(env, "high_level_previous_action", None)
    if action is None:
        return torch.zeros((env.num_envs, 11), device=env.device)
    return action


# -------------------------
# tcp / pose observations
# -------------------------
def ee_tcp_pose_w(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    ee_offset_pos=(0.06573, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return ee_tcp world pose [N,7] = [pos_w(3), quat_wxyz(4)]."""
    ee_pos, ee_quat = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    return torch.cat((ee_pos, ee_quat), dim=-1)


def base_to_doorway_center_b_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_z: float = 0.0,
) -> torch.Tensor:
    """Return doorway-center position relative to base, expressed in base frame XY."""
    robot: Articulation = env.scene[robot_cfg.name]
    center_w = _doorway_center_w(env, door_cfg, doorway_center_xy, doorway_z)
    return _world_vector_to_base(env, center_w - robot.data.root_pos_w, robot_cfg)[:, :2]


def last_high_base_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return previous deployable base command [vx, vy, wz]."""
    command = getattr(env, "high_level_previous_base_command", None)
    if command is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return command[:, :3]


def last_arm_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return previous high-level arm action only, excluding the 5-D base action prefix."""
    action = _get_action_history(env)
    if action.shape[-1] < 11:
        out = torch.zeros((env.num_envs, 6), device=env.device)
        available = max(int(action.shape[-1]) - 5, 0)
        if available > 0:
            out[:, :available] = action[:, 5:]
        return out
    return action[:, 5:11]


def high_base_command_3d(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current deployable base command [vx, vy, wz]."""
    command = getattr(env, "high_level_base_command", None)
    if command is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return command[:, :3]


def base_velocity_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return base velocity [vx_b, vy_b, wx_b, wy_b, wz_b]."""
    robot: Articulation = env.scene[robot_cfg.name]
    if hasattr(robot.data, "root_lin_vel_b"):
        lin_vel_b = robot.data.root_lin_vel_b
    else:
        lin_vel_b = quat_rotate(quat_conjugate(robot.data.root_quat_w), robot.data.root_lin_vel_w)
    if hasattr(robot.data, "root_ang_vel_b"):
        ang_vel_b = robot.data.root_ang_vel_b
    else:
        ang_vel_b = quat_rotate(quat_conjugate(robot.data.root_quat_w), robot.data.root_ang_vel_w)
    return torch.cat((lin_vel_b[:, :2], ang_vel_b), dim=-1)


def projected_gravity_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return gravity direction projected in base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.projected_gravity_b


def base_height(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return base/root height in world z."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_pos_w[:, 2:3]


def doorway_forward_axis_b_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    door_cfg: SceneEntityCfg = SceneEntityCfg("door"),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return door-frame doorway forward axis expressed in base frame XY."""
    axis_w = _doorway_forward_w(env, door_cfg, doorway_forward_axis)
    axis_b_xy = _world_vector_to_base(env, axis_w, robot_cfg)[:, :2]
    return axis_b_xy / (torch.norm(axis_b_xy, dim=-1, keepdim=True) + eps)


def door_panel_forward_axis_b_xy(
    env: ManagerBasedRLEnv,
    panel_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    panel_forward_axis: tuple[float, float, float] = (0.0, -1.0, 0.0),
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Return the moving door-panel forward axis in robot-base XY coordinates."""
    _, panel_quat_w = _get_single_body_pose_w(env, panel_cfg)
    axis_panel = _expand_vec3(panel_forward_axis, panel_quat_w[..., :3])
    axis_w = quat_rotate(panel_quat_w, axis_panel)
    axis_b_xy = _world_vector_to_base(env, axis_w, robot_cfg)[:, :2]
    return axis_b_xy / (torch.norm(axis_b_xy, dim=-1, keepdim=True) + eps)


def door_unlock_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Physical unlock state as a deployable-independent privileged scalar [N, 1]."""
    if hasattr(env, "_door_lock_mode"):
        unlocked = env._door_lock_mode == 2
    elif hasattr(env, "_door_unlocked"):
        unlocked = env._door_unlocked.to(dtype=torch.bool)
    else:
        unlocked = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return unlocked.float().unsqueeze(-1)


def door_stage_id(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Current gated task stage (0 approach, 1 unlock, 2 push) as [N, 1]."""
    grasped = getattr(env, "_grasp_success_given", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    unlocked = door_unlock_state(env).squeeze(-1).to(dtype=torch.bool)
    stage = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    stage = torch.where(grasped.to(dtype=torch.bool) & (~unlocked), torch.ones_like(stage), stage)
    stage = torch.where(unlocked, torch.full_like(stage, 2.0), stage)
    return stage.unsqueeze(-1)


def handle_target_point_b(
    env: ManagerBasedRLEnv,
    handle_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handle_offset_h=(0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return handle target point position relative to base, expressed in base frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    target_w = _handle_target_point_w(env, handle_cfg, handle_offset_h)
    return _world_vector_to_base(env, target_w - robot.data.root_pos_w, robot_cfg)


def ee_to_handle_target_b(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    handle_offset_h=(0.0, 0.0, 0.0),
    ee_offset_pos=(0.0, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return vector from EE to handle target point, expressed in base frame."""
    target_b = handle_target_point_b(env, handle_cfg, robot_cfg, handle_offset_h)
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos_w, _ = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    ee_b = _world_vector_to_base(env, ee_pos_w - robot.data.root_pos_w, robot_cfg)
    return target_b - ee_b


def ee_pos_in_handle_frame(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos=(0.06573, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return ee_tcp position error expressed in handle frame: [N,3]."""
    ee_pos, _ = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    h_pos, h_quat = _get_single_body_pose_w(env, handle_cfg)

    delta_w = ee_pos - h_pos
    delta_h = quat_rotate(quat_conjugate(h_quat), delta_w)
    return delta_h


def ee_quat_error_handle_frame(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos=(0.06573, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Return tcp orientation error q_err = q_handle^-1 * q_tcp, [N,4]."""
    _, ee_quat = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    _, h_quat = _get_single_body_pose_w(env, handle_cfg)
    q_err = quat_mul(quat_conjugate(h_quat), ee_quat)
    return quat_normalize(q_err)


def finger_contact_norms(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "left_finger_contact",
    right_sensor_name: str = "right_finger_contact",
) -> torch.Tensor:
    """Return [N,2]: left/right contact force norms."""
    left: ContactSensor = env.scene[left_sensor_name]
    right: ContactSensor = env.scene[right_sensor_name]
    f_l = _contact_force_norm(left)
    f_r = _contact_force_norm(right)
    return torch.stack((f_l, f_r), dim=-1)


def gripper_opening(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    joint_name: str = "gripper_joint",
) -> torch.Tensor:
    """Return the configured gripper opening proxy [N,1] from active joint only."""
    robot: Articulation = env.scene[robot_cfg.name]
    jnames = robot.data.joint_names
    jid = jnames.index(joint_name)
    return robot.data.joint_pos[:, jid : jid + 1]


def gripper_width(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    finger_joint_names=("gripper_joint", "joint8"),
) -> torch.Tensor:
    """Optional generic width proxy for robots with finger joints."""
    robot: Articulation = env.scene[robot_cfg.name]
    jnames = robot.data.joint_names
    joint_ids = [jnames.index(name) for name in finger_joint_names if name in jnames]
    if len(joint_ids) == 0:
        raise RuntimeError(f"No finger joints found in {finger_joint_names}. Available joints: {jnames}")
    width = sum(robot.data.joint_pos[:, jid] for jid in joint_ids)
    return width.unsqueeze(-1)


def last_action(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...] = ("arm_action", "gripper_action"),
    action_dims: tuple[int, ...] = (6, 1),
) -> torch.Tensor:
    """Return previous policy action commands concatenated as [N, action_dim].

    This intentionally uses raw/processed policy commands rather than
    stage-scaled applied target increments. Use last_applied_arm_delta() for
    the actual arm position-target increment applied after action scaling.
    """
    action_terms = []
    action_manager = getattr(env, "action_manager", None)

    for idx, name in enumerate(action_names):
        term = None
        if action_manager is not None and hasattr(action_manager, "_terms") and name in action_manager._terms:
            term = action_manager._terms[name]
        elif action_manager is not None and hasattr(action_manager, "get_term"):
            try:
                term = action_manager.get_term(name)
            except Exception:
                term = None

        if term is None:
            dim = int(action_dims[idx]) if idx < len(action_dims) else 0
            if dim > 0:
                action_terms.append(torch.zeros((env.num_envs, dim), device=env.device))
            continue

        if hasattr(term, "raw_actions"):
            action = term.raw_actions
        elif hasattr(term, "processed_actions"):
            action = term.processed_actions
        else:
            continue

        if action.ndim == 1:
            action = action.unsqueeze(-1)
        action_terms.append(action)

    if len(action_terms) == 0:
        fallback_dim = int(sum(action_dims))
        return torch.zeros((env.num_envs, fallback_dim), device=env.device)

    out = torch.cat(action_terms, dim=-1)
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["observations/last_action_abs_mean"] = out.abs().mean().detach()
    return out


def _get_action_term(env: ManagerBasedRLEnv, action_name: str):
    action_manager = getattr(env, "action_manager", None)
    if action_manager is None:
        return None
    if hasattr(action_manager, "_terms") and action_name in action_manager._terms:
        return action_manager._terms[action_name]
    if hasattr(action_manager, "get_term"):
        try:
            return action_manager.get_term(action_name)
        except Exception:
            return None
    return None


def last_applied_arm_delta(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Return previous arm position-target increment after clamp and stage scaling."""
    term = _get_action_term(env, action_name)
    if term is None or not hasattr(term, "applied_delta"):
        return torch.zeros((env.num_envs, 6), device=env.device)

    out = term.applied_delta

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["observations/last_applied_arm_delta_abs_mean"] = out.abs().mean().detach()

    return out


def arm_q_des_error(
    env: ManagerBasedRLEnv,
    action_name: str = "arm_action",
) -> torch.Tensor:
    """Return arm position-target tracking error q_des - q.

    Shape: [N, 6]. This is deployable low-level controller state.
    """
    term = _get_action_term(env, action_name)
    if term is None or not hasattr(term, "q_des") or not hasattr(term, "joint_ids"):
        return torch.zeros((env.num_envs, 6), device=env.device)

    robot: Articulation = env.scene["robot"]
    q = robot.data.joint_pos[:, term.joint_ids]
    err = term.q_des - q

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["observations/arm_q_des_error_abs_mean"] = err.abs().mean().detach()

    return err


# --- small helpers: axis-angle -> quat (wxyz) ---
def _axis_angle_to_quat(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def _rand_quat_noise(n: int, rot_std_rad: float, device) -> torch.Tensor:
    axis = torch.randn((n, 3), device=device)
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-8)
    angle = torch.randn((n,), device=device) * rot_std_rad
    return _axis_angle_to_quat(axis, angle)


def ee_pos_in_noisy_handle_frame(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos=(0.06573, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
    pos_std=0.01,
    rot_std_deg=5.0,
) -> torch.Tensor:
    """ee_tcp position error expressed in noisy handle frame: [N,3]."""
    ee_pos, _ = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    h_pos, h_quat = _get_single_body_pose_w(env, handle_cfg)

    device = ee_pos.device
    n = ee_pos.shape[0]

    h_pos_n = h_pos + torch.randn_like(h_pos) * pos_std
    rot_std_rad = rot_std_deg * torch.pi / 180.0
    dq = _rand_quat_noise(n, rot_std_rad, device)
    h_quat_n = quat_normalize(quat_mul(h_quat, dq))

    delta_w = ee_pos - h_pos_n
    delta_h = quat_rotate(quat_conjugate(h_quat_n), delta_w)
    return delta_h


def ee_quat_error_noisy_handle_frame(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    handle_cfg: SceneEntityCfg,
    ee_offset_pos=(0.06573, 0.0, 0.0),
    ee_offset_quat=(1.0, 0.0, 0.0, 0.0),
    rot_std_deg=5.0,
) -> torch.Tensor:
    """q_err = q_handle(noisy)^-1 * q_tcp, [N,4]."""
    _, ee_quat = _get_single_body_pose_w_with_offset(
        env, ee_cfg, offset_pos=ee_offset_pos, offset_quat=ee_offset_quat
    )
    _, h_quat = _get_single_body_pose_w(env, handle_cfg)

    device = ee_quat.device
    n = ee_quat.shape[0]
    rot_std_rad = rot_std_deg * torch.pi / 180.0
    dq = _rand_quat_noise(n, rot_std_rad, device)
    h_quat_n = quat_normalize(quat_mul(h_quat, dq))

    q_err = quat_mul(quat_conjugate(h_quat_n), ee_quat)
    return quat_normalize(q_err)
