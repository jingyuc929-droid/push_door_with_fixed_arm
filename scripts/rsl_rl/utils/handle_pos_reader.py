# --- GUI HUD: show handle_joint pos/vel from IsaacLab env scene data (no terminal prints) ---
import time
import omni.ui as ui
import omni.kit.app

def start_handle_joint_hud(env, joint_name="handle_joint", env_index=0, update_hz=20.0):
    door = env.scene["door"]  # 你的 SceneEntity 名字如果不是 door，改这里
    # resolve joint id once
    jnames = list(door.data.joint_names)
    if joint_name not in jnames:
        raise RuntimeError(f"Joint '{joint_name}' not found. Available: {jnames}")
    jid = jnames.index(joint_name)

    win = ui.Window("Handle Joint HUD (IsaacLab)", width=360, height=140)
    with win.frame:
        with ui.VStack(spacing=6):
            lbl_pos = ui.Label("pos: --")
            lbl_vel = ui.Label("vel: --")

    dt = 1.0 / max(1e-3, float(update_hz))
    last_t = 0.0

    def _on_update(_evt):
        nonlocal last_t
        now = time.time()
        if now - last_t < dt:
            return
        last_t = now
        pos = float(door.data.joint_pos[env_index, jid].item())
        vel = float(door.data.joint_vel[env_index, jid].item())
        lbl_pos.text = f"pos: {pos:.6f}"
        lbl_vel.text = f"vel: {vel:.6f}"

    stream = omni.kit.app.get_app().get_update_event_stream()
    sub = stream.create_subscription_to_pop(_on_update)
    return win, sub
