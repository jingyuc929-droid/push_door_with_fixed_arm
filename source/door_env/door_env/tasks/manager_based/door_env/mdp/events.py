import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def update_door_lock_hysteresis_delayed_release(
    env,
    env_ids,
    asset_cfg: SceneEntityCfg | None = None,
    handle_joint_name: str = "handle_joint",
    door_joint_name: str = "door_joint",
    unlock_threshold: float = -0.30,
    relock_threshold: float = -0.26,  # deprecated: relock now uses door + handle closed thresholds
    unlock_hold_steps: int = 4,
    release_delay_steps: int = 8,
    relock_door_threshold: float = 0.05,
    relock_handle_threshold: float = -0.03,
    door_closed_pos: float = 0.0,
    door_open_sign: float = 1.0,
    lock_door_pos: float = 0.0,
):
    LOCKED = 0
    UNLOCK_DELAY = 1
    UNLOCKED = 2

    door_name = asset_cfg.name if isinstance(asset_cfg, SceneEntityCfg) else "door"
    door: Articulation = env.scene[door_name]
    device = env.device
    N = env.num_envs

    joint_names = list(door.data.joint_names)
    handle_jid = joint_names.index(handle_joint_name)
    door_jid = joint_names.index(door_joint_name)

    handle_pos = door.data.joint_pos[:, handle_jid]
    door_pos = door.data.joint_pos[:, door_jid]
    door_open = float(door_open_sign) * (door_pos - float(door_closed_pos))
    door_open = torch.clamp(door_open, min=0.0)

    # ------------------------------------------------------------
    # buffers
    # ------------------------------------------------------------
    if (not hasattr(env, "_door_lock_mode")) or (env._door_lock_mode.shape[0] != N):
        if hasattr(env, "_door_unlocked") and env._door_unlocked.shape[0] == N:
            env._door_lock_mode = torch.where(
                env._door_unlocked.to(dtype=torch.bool),
                torch.full((N,), UNLOCKED, dtype=torch.int32, device=device),
                torch.zeros(N, dtype=torch.int32, device=device),
            )
        else:
            env._door_lock_mode = torch.zeros(N, dtype=torch.int32, device=device)

    if (not hasattr(env, "_door_unlock_counter")) or (env._door_unlock_counter.shape[0] != N):
        env._door_unlock_counter = torch.zeros(N, dtype=torch.int32, device=device)

    if (not hasattr(env, "_door_release_delay_counter")) or (env._door_release_delay_counter.shape[0] != N):
        env._door_release_delay_counter = torch.zeros(N, dtype=torch.int32, device=device)

    if (not hasattr(env, "_door_unlocked")) or (env._door_unlocked.shape[0] != N):
        env._door_unlocked = torch.zeros(N, dtype=torch.bool, device=device)

    if (not hasattr(env, "_door_lock_state")) or (env._door_lock_state.shape[0] != N):
        env._door_lock_state = torch.zeros(N, dtype=torch.bool, device=device)

    # ------------------------------------------------------------
    # reset detection
    # ------------------------------------------------------------
    reset_mask = None
    if hasattr(env, "reset_buf"):
        try:
            reset_mask = env.reset_buf.to(dtype=torch.bool)
        except Exception:
            reset_mask = None
    if reset_mask is None and hasattr(env, "episode_length_buf"):
        if (not hasattr(env, "_prev_ep_len_door_lock")) or (env._prev_ep_len_door_lock.shape[0] != N):
            env._prev_ep_len_door_lock = env.episode_length_buf.clone()
        reset_mask = (env.episode_length_buf < env._prev_ep_len_door_lock) | (env.episode_length_buf == 0)
        env._prev_ep_len_door_lock = env.episode_length_buf.clone()

    if reset_mask is not None:
        env._door_lock_mode = torch.where(
            reset_mask,
            torch.zeros_like(env._door_lock_mode),
            env._door_lock_mode,
        )
        env._door_unlock_counter = torch.where(
            reset_mask,
            torch.zeros_like(env._door_unlock_counter),
            env._door_unlock_counter,
        )
        env._door_release_delay_counter = torch.where(
            reset_mask,
            torch.zeros_like(env._door_release_delay_counter),
            env._door_release_delay_counter,
        )
        env._door_unlocked = torch.where(
            reset_mask,
            torch.zeros_like(env._door_unlocked),
            env._door_unlocked,
        )
        env._door_lock_state = torch.where(
            reset_mask,
            torch.zeros_like(env._door_lock_state),
            env._door_lock_state,
        )

    mode_start = env._door_lock_mode.clone()
    locked_mask = mode_start == LOCKED
    delay_mask = mode_start == UNLOCK_DELAY
    unlocked_mask = mode_start == UNLOCKED

    deep_press = handle_pos < float(unlock_threshold)
    env._door_unlock_counter = torch.where(
        locked_mask & deep_press,
        env._door_unlock_counter + 1,
        torch.where(locked_mask, torch.zeros_like(env._door_unlock_counter), env._door_unlock_counter),
    )

    enter_delay = locked_mask & (env._door_unlock_counter >= int(max(1, unlock_hold_steps)))
    env._door_lock_mode = torch.where(
        enter_delay,
        torch.full_like(env._door_lock_mode, UNLOCK_DELAY),
        env._door_lock_mode,
    )
    env._door_release_delay_counter = torch.where(
        enter_delay,
        torch.full_like(env._door_release_delay_counter, int(max(0, release_delay_steps))),
        env._door_release_delay_counter,
    )
    env._door_unlock_counter = torch.where(
        enter_delay,
        torch.zeros_like(env._door_unlock_counter),
        env._door_unlock_counter,
    )

    # ------------------------------------------------------------
    # UNLOCK_DELAY deterministically transitions to UNLOCKED.
    # ------------------------------------------------------------
    env._door_release_delay_counter = torch.where(
        delay_mask,
        torch.clamp(env._door_release_delay_counter - 1, min=0),
        env._door_release_delay_counter,
    )
    finish_delay = delay_mask & (env._door_release_delay_counter <= 0)
    env._door_lock_mode = torch.where(
        finish_delay,
        torch.full_like(env._door_lock_mode, UNLOCKED),
        env._door_lock_mode,
    )

    # ------------------------------------------------------------
    # UNLOCKED relocks only when both door and handle are near closed.
    # ------------------------------------------------------------
    relock_ok = (
        unlocked_mask
        & (door_open < float(relock_door_threshold))
        & (handle_pos > float(relock_handle_threshold))
    )
    env._door_lock_mode = torch.where(
        relock_ok,
        torch.zeros_like(env._door_lock_mode),
        env._door_lock_mode,
    )
    env._door_unlock_counter = torch.where(
        relock_ok,
        torch.zeros_like(env._door_unlock_counter),
        env._door_unlock_counter,
    )
    env._door_release_delay_counter = torch.where(
        relock_ok,
        torch.zeros_like(env._door_release_delay_counter),
        env._door_release_delay_counter,
    )

    # ------------------------------------------------------------
    # physical door locking: lock only envs that started this call locked/delay.
    # ------------------------------------------------------------
    physically_locked = locked_mask | delay_mask

    if physically_locked.any():
        qpos = door.data.joint_pos.clone()
        qvel = door.data.joint_vel.clone()

        qpos[physically_locked, door_jid] = float(lock_door_pos)
        qvel[physically_locked, door_jid] = 0.0

        door.write_joint_state_to_sim(qpos, qvel)

    # Compatibility buffers: True means unlocked/free in current stage logic.
    env._door_unlocked[:] = env._door_lock_mode == UNLOCKED
    env._door_lock_state[:] = env._door_unlocked

    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["door_lock/mode_locked_ratio"] = (env._door_lock_mode == LOCKED).float().mean().detach()
    env.extras["log"]["door_lock/mode_delay_ratio"] = (env._door_lock_mode == UNLOCK_DELAY).float().mean().detach()
    env.extras["log"]["door_lock/mode_unlocked_ratio"] = (env._door_lock_mode == UNLOCKED).float().mean().detach()
    env.extras["log"]["door_lock/unlock_counter_mean"] = env._door_unlock_counter.float().mean().detach()
    env.extras["log"]["door_lock/release_delay_counter_mean"] = (
        env._door_release_delay_counter.float().mean().detach()
    )
    env.extras["log"]["door_lock/door_open_mean"] = door_open.mean().detach()
    env.extras["log"]["door_lock/door_open_max"] = door_open.max().detach()
    env.extras["log"]["door_lock/handle_pos_mean"] = handle_pos.mean().detach()
    env.extras["log"]["door_lock/handle_pos_min"] = handle_pos.min().detach()
    env.extras["log"]["door_lock/relock_ok_ratio"] = relock_ok.float().mean().detach()


def visualize_doorway_debug(
    env,
    env_ids,
    doorway_center_xy: tuple[float, float] = (0.0, 0.0),
    doorway_forward_axis: tuple[float, float] = (1.0, 0.0),
    z: float = 0.05,
    arrow_length: float = 0.5,
    point_size: float = 18.0,
    line_width: float = 5.0,
):
    """Draw doorway center and forward axis in GUI for quick reward-frame inspection."""
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except Exception:
        return

    draw = _debug_draw.acquire_debug_draw_interface()
    if not hasattr(env, "_doorway_debug_draw_initialized"):
        env._doorway_debug_draw_initialized = True
        env._doorway_debug_draw_counter = 0

    # Keep this overlay tidy when running zero_agent for a long time.
    try:
        draw.clear_points()
        draw.clear_lines()
    except Exception:
        pass

    device = env.device
    dtype = torch.float32
    center_xy = torch.tensor(doorway_center_xy, device=device, dtype=dtype)
    forward_xy = torch.tensor(doorway_forward_axis, device=device, dtype=dtype)
    forward_xy = forward_xy / torch.clamp(torch.linalg.norm(forward_xy), min=1.0e-6)

    center = (float(center_xy[0]), float(center_xy[1]), float(z))
    end_xy = center_xy + float(arrow_length) * forward_xy
    end = (float(end_xy[0]), float(end_xy[1]), float(z))

    # Red point: doorway center. Blue line + two small wings: forward direction.
    draw.draw_points([center], [(1.0, 0.0, 0.0, 1.0)], [float(point_size)])
    starts = [center]
    ends = [end]
    colors = [(0.0, 0.25, 1.0, 1.0)]
    widths = [float(line_width)]

    left = torch.tensor([-forward_xy[1], forward_xy[0]], device=device, dtype=dtype)
    wing_len = 0.18 * float(arrow_length)
    wing_back = 0.25 * float(arrow_length)
    wing_base = end_xy - wing_back * forward_xy
    wing_a = wing_base + wing_len * left
    wing_b = wing_base - wing_len * left
    starts.extend([end, end])
    ends.extend(
        [
            (float(wing_a[0]), float(wing_a[1]), float(z)),
            (float(wing_b[0]), float(wing_b[1]), float(z)),
        ]
    )
    colors.extend([(0.0, 0.25, 1.0, 1.0), (0.0, 0.25, 1.0, 1.0)])
    widths.extend([float(line_width), float(line_width)])
    draw.draw_lines(starts, ends, colors, widths)

    env._doorway_debug_draw_counter += 1
    if not hasattr(env, "extras") or env.extras is None:
        env.extras = {}
    if "log" not in env.extras:
        env.extras["log"] = {}
    env.extras["log"]["doorway_debug/center_x"] = center_xy[0].detach()
    env.extras["log"]["doorway_debug/center_y"] = center_xy[1].detach()
