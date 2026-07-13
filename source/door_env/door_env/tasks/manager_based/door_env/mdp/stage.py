import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

# -------------------------
# 1) snapshot / apply
# -------------------------
def _snapshot_articulation(asset: Articulation, env_ids: torch.Tensor):
    root = asset.data.root_state_w[env_ids].clone()
    qpos = asset.data.joint_pos[env_ids].clone()
    qvel = asset.data.joint_vel[env_ids].clone()
    return root, qpos, qvel


def reset_root_state_to_default(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
):
    """Reset an articulation root pose/velocity to its configured initial state."""
    asset: Articulation = env.scene[asset_cfg.name]

    if hasattr(asset.data, "default_root_state"):
        root_state = asset.data.default_root_state[env_ids].clone()
        if hasattr(env.scene, "env_origins"):
            root_state[:, :3] += env.scene.env_origins[env_ids]
    else:
        root_state = asset.data.root_state_w[env_ids].clone()
        root_state[:, 7:] = 0.0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)


def _apply_articulation(
    asset: Articulation,
    env_ids: torch.Tensor,
    root,
    qpos,
    qvel,
    apply_root: bool = False,
):
    # 当前 door / robot 都是固定基座，默认只回放 joint 更稳
    if apply_root:
        asset.write_root_state_to_sim(root, env_ids=env_ids)
    asset.write_joint_state_to_sim(qpos, qvel, env_ids=env_ids)


# -------------------------
# 2) small helpers
# -------------------------
def _ensure_bool_buffer(env, name: str):
    if not hasattr(env, name):
        setattr(env, name, torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    return getattr(env, name)


def _sync_door_lock_state(env, env_ids: torch.Tensor, unlocked_values: torch.Tensor):
    # 当前按与你现有逻辑最接近的方式，同步两个状态。
    # 如果你的 _door_lock_state 在别处表示 "locked=True"，把下一行改成 ~unlocked_values 即可。
    if hasattr(env, "_door_unlocked"):
        env._door_unlocked[env_ids] = unlocked_values
    if hasattr(env, "_door_lock_state"):
        env._door_lock_state[env_ids] = unlocked_values
    if hasattr(env, "_door_lock_mode"):
        env._door_lock_mode[env_ids] = torch.where(
            unlocked_values.to(dtype=torch.bool),
            torch.full_like(env._door_lock_mode[env_ids], 2),
            torch.zeros_like(env._door_lock_mode[env_ids]),
        )
    if hasattr(env, "_door_unlock_counter"):
        env._door_unlock_counter[env_ids] = 0
    if hasattr(env, "_door_release_delay_counter"):
        env._door_release_delay_counter[env_ids] = 0


def _clear_reward_phase_buffers(env, env_ids: torch.Tensor):
    # bool latches
    for name in [
        "_grasp_success_given",
        "_unlock_success_given",
        "_unlock_reached_once",
        "_keep_all_phase",
        "_keep_all_prev_keep_ok",
        "_unlock_crossed_once",
        "_prev_grasp_success_given",
        "_door_released",
    ]:
        if hasattr(env, name):
            buf = getattr(env, name)
            if isinstance(buf, torch.Tensor) and buf.shape[0] == env.num_envs and buf.dtype == torch.bool:
                buf[env_ids] = False

    # int counters
    for name in [
        "_grasp_success_counter",
        "_unlock_success_counter",
        "_grasp_hold_counter",
        "_grasp_recent_ttl",
        "_grasp_recent_grace",
        "_unlock_reset_cooldown",
        "_door_lock_mode",
        "_door_unlock_counter",
        "_door_release_delay_counter",
        "_door_unlock_hold_counter",
    ]:
        if hasattr(env, name):
            buf = getattr(env, name)
            if isinstance(buf, torch.Tensor) and buf.shape[0] == env.num_envs and buf.dtype in (torch.int32, torch.int64):
                buf[env_ids] = 0

    # float EMA / history buffers
    for name in [
        "_unlock_prog_ema",
        "_press_prev_handle_pos",
        "_press_vel_ema",
        "_door_open_ema",
        "_keep_all_door_open_ema",
    ]:
        if hasattr(env, name):
            buf = getattr(env, name)
            if isinstance(buf, torch.Tensor) and buf.shape[0] == env.num_envs and torch.is_floating_point(buf):
                buf[env_ids] = 0.0


# -------------------------
# 3) ring buffer archive
# -------------------------
def _init_archive(env, name: str, cap: int, robot: Articulation, door: Articulation):
    dev = env.device
    env.__dict__[name] = {
        "cap": cap,
        "ptr": 0,
        "size": 0,
        "robot_root": torch.zeros((cap, robot.data.root_state_w.shape[1]), device=dev),
        "robot_qpos": torch.zeros((cap, robot.data.joint_pos.shape[1]), device=dev),
        "robot_qvel": torch.zeros((cap, robot.data.joint_vel.shape[1]), device=dev),
        "door_root": torch.zeros((cap, door.data.root_state_w.shape[1]), device=dev),
        "door_qpos": torch.zeros((cap, door.data.joint_pos.shape[1]), device=dev),
        "door_qvel": torch.zeros((cap, door.data.joint_vel.shape[1]), device=dev),
        # stage flags
        "door_unlocked": torch.zeros((cap,), dtype=torch.bool, device=dev),
        "grasp_success": torch.zeros((cap,), dtype=torch.bool, device=dev),
        "unlock_success": torch.zeros((cap,), dtype=torch.bool, device=dev),
    }


def push_archive_from_env(
    env,
    name: str,
    env_ids: torch.Tensor,
    cap: int,
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg,
    store_unlock_flag: bool = False,
):
    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]

    if name not in env.__dict__:
        _init_archive(env, name, cap, robot, door)

    A = env.__dict__[name]
    r_root, r_qpos, r_qvel = _snapshot_articulation(robot, env_ids)
    d_root, d_qpos, d_qvel = _snapshot_articulation(door, env_ids)

    k = env_ids.shape[0]
    idx = (torch.arange(k, device=env.device) + A["ptr"]) % A["cap"]

    A["robot_root"][idx] = r_root
    A["robot_qpos"][idx] = r_qpos
    A["robot_qvel"][idx] = r_qvel
    A["door_root"][idx] = d_root
    A["door_qpos"][idx] = d_qpos
    A["door_qvel"][idx] = d_qvel

    if hasattr(env, "_door_unlocked"):
        A["door_unlocked"][idx] = env._door_unlocked[env_ids]
    else:
        A["door_unlocked"][idx] = False

    if hasattr(env, "_grasp_success_given"):
        A["grasp_success"][idx] = env._grasp_success_given[env_ids]
    else:
        A["grasp_success"][idx] = False

    if hasattr(env, "_unlock_success_given"):
        A["unlock_success"][idx] = env._unlock_success_given[env_ids]
    else:
        A["unlock_success"][idx] = A["door_unlocked"][idx] if store_unlock_flag else False

    A["ptr"] = int((A["ptr"] + k) % A["cap"])
    A["size"] = int(min(A["cap"], A["size"] + k))


def sample_archive_to_env(
    env,
    name: str,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg,
    restore_stage_flags: bool = False,
):
    A = env.__dict__.get(name, None)
    if (A is None) or (A["size"] <= 0):
        return False

    robot: Articulation = env.scene[robot_cfg.name]
    door: Articulation = env.scene[door_cfg.name]

    rows = torch.randint(0, A["size"], (env_ids.shape[0],), device=env.device)

    _apply_articulation(robot, env_ids, A["robot_root"][rows], A["robot_qpos"][rows], A["robot_qvel"][rows])
    _apply_articulation(door, env_ids, A["door_root"][rows], A["door_qpos"][rows], A["door_qvel"][rows])

    if restore_stage_flags:
        _ensure_bool_buffer(env, "_door_unlocked")
        _ensure_bool_buffer(env, "_door_lock_state")
        _ensure_bool_buffer(env, "_grasp_success_given")
        _ensure_bool_buffer(env, "_unlock_success_given")

        # 先清历史 buffer，再恢复 stage flags，避免旧 episode 内存污染
        _clear_reward_phase_buffers(env, env_ids)

        env._grasp_success_given[env_ids] = A["grasp_success"][rows]
        env._unlock_success_given[env_ids] = A["unlock_success"][rows]
        _sync_door_lock_state(env, env_ids, A["door_unlocked"][rows])

    return True


# -------------------------
# 4) staged reset event term
# -------------------------
def staged_reset_from_archive(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    door_cfg: SceneEntityCfg,
    p_grasp_start: float = 0.6,
    p_unlock_start: float = 0.6,
    min_archive: int = 32,
    cap_grasp: int = 512,
    cap_unlock: int = 512,
):
    u = torch.rand((env_ids.shape[0],), device=env.device)

    use_unlock = u < p_unlock_start
    use_grasp = (u >= p_unlock_start) & (u < (p_unlock_start + p_grasp_start))

    restored = torch.zeros(env_ids.shape[0], dtype=torch.bool, device=env.device)

    # unlock-stage restore
    if use_unlock.any():
        A = env.__dict__.get("_archive_unlock", None)
        if (A is not None) and (A["size"] >= min_archive):
            ok = sample_archive_to_env(
                env,
                "_archive_unlock",
                env_ids[use_unlock],
                robot_cfg,
                door_cfg,
                restore_stage_flags=True,
            )
            if ok:
                restored[use_unlock] = True

    # grasp-stage restore
    if use_grasp.any():
        A = env.__dict__.get("_archive_grasp", None)
        if (A is not None) and (A["size"] >= min_archive):
            ok = sample_archive_to_env(
                env,
                "_archive_grasp",
                env_ids[use_grasp],
                robot_cfg,
                door_cfg,
                restore_stage_flags=True,
            )
            if ok:
                restored[use_grasp] = True

                # grasp archive 恢复后，door 必须明确保持 locked / not-unlocked
                if hasattr(env, "_unlock_success_given"):
                    env._unlock_success_given[env_ids[use_grasp]] = False
                _sync_door_lock_state(
                    env,
                    env_ids[use_grasp],
                    torch.zeros(env_ids[use_grasp].shape[0], dtype=torch.bool, device=env.device),
                )

    # fallback: 没被 archive 恢复的 env，把门 joint 清零，并清理阶段状态
    # 注意：robot 侧仍依赖外部 reset_robot_joints 先执行
    fallback_ids = env_ids[~restored]
    if fallback_ids.numel() > 0:
        _ensure_bool_buffer(env, "_door_unlocked")
        _ensure_bool_buffer(env, "_door_lock_state")
        _ensure_bool_buffer(env, "_grasp_success_given")
        _ensure_bool_buffer(env, "_unlock_success_given")

        _clear_reward_phase_buffers(env, fallback_ids)

        env._grasp_success_given[fallback_ids] = False
        env._unlock_success_given[fallback_ids] = False
        _sync_door_lock_state(
            env,
            fallback_ids,
            torch.zeros(fallback_ids.shape[0], dtype=torch.bool, device=env.device),
        )

        door: Articulation = env.scene[door_cfg.name]
        qpos = door.data.joint_pos[fallback_ids].clone()
        qvel = torch.zeros_like(door.data.joint_vel[fallback_ids])

        qpos[:] = 0.0  # door_joint = 0, handle_joint = 0
        door.write_joint_state_to_sim(qpos, qvel, env_ids=fallback_ids)
