import argparse
import math
import sys
import time
from dataclasses import dataclass

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Scripted gripper symmetry test in Isaac Lab/Sim.")
parser.add_argument("--task", type=str, required=True, help="Gym task name.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="Hydra agent cfg entry point (unused, required for task cfg resolution).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time if possible.")
parser.add_argument("--video", action="store_true", default=False, help="Enable cameras for video if needed.")

# Script parameters
parser.add_argument("--open-seconds", type=float, default=1.0, help="Hold-open duration for each open phase.")
parser.add_argument("--close-seconds", type=float, default=1.0, help="Hold-close duration for each close phase.")
parser.add_argument("--post-rotate-open-seconds", type=float, default=0.5, help="Extra hold-open duration immediately after rotation.")
parser.add_argument("--rot-target-deg", type=float, default=90.0, help="Target wrist rotation in degrees.")
parser.add_argument("--rot-max-seconds", type=float, default=3.0, help="Max duration allowed for the rotate phase.")
parser.add_argument("--rot-action-dim", type=int, default=5, help="Which arm action dim to drive during wrist rotation (default: 5 for a 6D IK action).")
parser.add_argument("--rot-action-mag", type=float, default=0.8, help="Magnitude of rotation action during rotate phase.")
parser.add_argument("--rot-sign", type=float, default=1.0, help="Sign of wrist rotation action (+1 or -1).")
parser.add_argument("--gripper-open-action", type=float, default=1.0, help="Action value used to open the gripper.")
parser.add_argument("--gripper-close-action", type=float, default=-1.0, help="Action value used to close the gripper.")
parser.add_argument("--stats-open-width", type=float, default=0.088, help="Open width used for q-based/geometric closedness.")
parser.add_argument("--hud-update-hz", type=float, default=20.0, help="HUD refresh rate.")
parser.add_argument("--log-every", type=int, default=20, help="Print compact stats every N sim steps.")

# Scene / body names
parser.add_argument("--robot-name", type=str, default="robot")
parser.add_argument("--door-name", type=str, default="door")
parser.add_argument("--hand-body", type=str, default="link6")
parser.add_argument("--left-body", type=str, default="left_pad")
parser.add_argument("--right-body", type=str, default="right_pad")
parser.add_argument("--left-body-fallback", type=str, default="link7")
parser.add_argument("--right-body-fallback", type=str, default="link8")
parser.add_argument("--gripper-joint", type=str, default="gripper_joint")
parser.add_argument("--left-contact-sensor", type=str, default="left_finger_contact")
parser.add_argument("--right-contact-sensor", type=str, default="right_finger_contact")
parser.add_argument("--open-axis-hand", type=str, default="0,1,0", help="Hand-frame opening axis as comma-separated vector, e.g. '0,1,0'.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import omni.kit.app
import omni.ui as ui

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_tasks.utils.hydra import hydra_task_config
import door_env.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401


def parse_vec3(text: str) -> torch.Tensor:
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 3:
        raise ValueError(f"Expected 3 numbers for vec3, got: {text}")
    return torch.tensor(vals, dtype=torch.float32)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.stack((q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]), dim=-1)


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


def safe_unit(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)


def quat_relative_angle_deg(q_ref: torch.Tensor, q_cur: torch.Tensor) -> float:
    # Angle of relative quaternion, robust to sign flip.
    dot = torch.abs(torch.sum(q_ref * q_cur, dim=-1)).clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angle).item())


def resolve_action_dim(env) -> int:
    base_env = env.unwrapped
    if hasattr(base_env, "num_actions"):
        return int(base_env.num_actions)
    if hasattr(base_env, "action_manager") and hasattr(base_env.action_manager, "total_action_dim"):
        return int(base_env.action_manager.total_action_dim)
    shape = getattr(env.action_space, "shape", None)
    if shape is None:
        raise RuntimeError("Cannot resolve action dimension.")
    return int(shape[-1])


def find_body_id(names, preferred: str, fallback: str | None = None) -> int:
    if preferred in names:
        return names.index(preferred)
    if fallback is not None and fallback in names:
        return names.index(fallback)
    raise RuntimeError(f"Cannot find body '{preferred}' (fallback '{fallback}') in {names}")


def filtered_force_norm(sensor, num_envs: int, device) -> torch.Tensor:
    if sensor is None:
        return torch.zeros((num_envs,), device=device)
    fm = getattr(sensor.data, "force_matrix_w", None)
    if fm is None:
        return torch.zeros((num_envs,), device=device)
    if fm.ndim == 4:
        vec = fm.sum(dim=2)  # [N,B,3]
        mag = torch.linalg.norm(vec, dim=-1)
        return mag.max(dim=1).values
    if fm.ndim == 3:
        vec = fm.sum(dim=1)
        return torch.linalg.norm(vec, dim=-1)
    return torch.zeros((num_envs,), device=device)


@dataclass
class Phase:
    name: str
    kind: str  # 'hold' or 'rotate'
    gripper_action: float
    duration_steps: int = 0


class ScriptState:
    def __init__(self, phases: list[Phase]):
        self.phases = phases
        self.phase_idx = 0
        self.phase_step = 0
        self.rotate_q_ref = None
        self.rotate_angle_deg = 0.0
        self.rotate_done = False

    @property
    def phase(self) -> Phase:
        return self.phases[min(self.phase_idx, len(self.phases) - 1)]

    @property
    def is_finished(self) -> bool:
        return self.phase_idx >= len(self.phases)

    def advance(self):
        self.phase_idx += 1
        self.phase_step = 0
        self.rotate_q_ref = None
        self.rotate_angle_deg = 0.0
        self.rotate_done = False


class SymmetryHUD:
    def __init__(self, get_stats_fn, script_state: ScriptState, update_hz: float = 20.0):
        self._get_stats_fn = get_stats_fn
        self._script_state = script_state
        self._update_dt = 1.0 / max(1e-3, float(update_hz))
        self._last_t = 0.0
        self._win = ui.Window("Gripper Symmetry Test HUD", width=520, height=420)
        with self._win.frame:
            with ui.VStack(spacing=4):
                self.lbl_phase = ui.Label("phase: --")
                self.lbl_rot = ui.Label("rot_progress: --")
                self.lbl_q = ui.Label("gripper_joint q(env0): --")
                self.lbl_width_drv = ui.Label("derived width=2*q (env0): --")
                self.lbl_gap_proj = ui.Label("gap_proj(open-axis, env0): --")
                self.lbl_gap_euc = ui.Label("gap_euclid(env0): --")
                self.lbl_closed_drv = ui.Label("closed_drv(env0): --")
                self.lbl_closed_geom = ui.Label("closed_geom(env0): --")
                self.lbl_closed_mean_drv = ui.Label("closed_mean_drv(all envs): --")
                self.lbl_closed_mean_geom = ui.Label("closed_mean_geom(all envs): --")
                self.lbl_proj_lr = ui.Label("left/right proj(env0): -- / --")
                self.lbl_asym = ui.Label("asym_proj(env0): --")
                self.lbl_force = ui.Label("|F| L/R (env0): -- / --")
        self._sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self._on_update)

    def _on_update(self, _evt):
        now = time.time()
        if now - self._last_t < self._update_dt:
            return
        self._last_t = now
        try:
            s = self._get_stats_fn()
            self.lbl_phase.text = f"phase: {s['phase']}   step_in_phase: {s['phase_step']}"
            self.lbl_rot.text = f"rot_progress: {s['rot_deg']:.1f} / {s['rot_target_deg']:.1f} deg"
            self.lbl_q.text = f"gripper_joint q(env0): {s['q_env0']:.5f}"
            self.lbl_width_drv.text = f"derived width=2*q (env0): {s['width_drv_env0']:.5f}"
            self.lbl_gap_proj.text = f"gap_proj(open-axis, env0): {s['gap_proj_env0']:.5f}"
            self.lbl_gap_euc.text = f"gap_euclid(env0): {s['gap_euc_env0']:.5f}"
            self.lbl_closed_drv.text = f"closed_drv(env0): {s['closed_drv_env0']:.3f}"
            self.lbl_closed_geom.text = f"closed_geom(env0): {s['closed_geom_env0']:.3f}"
            self.lbl_closed_mean_drv.text = f"closed_mean_drv(all envs): {s['closed_mean_drv']:.3f}"
            self.lbl_closed_mean_geom.text = f"closed_mean_geom(all envs): {s['closed_mean_geom']:.3f}"
            self.lbl_proj_lr.text = f"left/right proj(env0): {s['left_proj_env0']:+.5f} / {s['right_proj_env0']:+.5f}"
            self.lbl_asym.text = f"asym_proj(env0): {s['asym_env0']:.5f}"
            self.lbl_force.text = f"|F| L/R (env0): {s['fL_env0']:.3f} / {s['fR_env0']:.3f}"
        except Exception as e:
            self.lbl_phase.text = f"HUD error: {e}"


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else getattr(env_cfg, "seed", None)
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped
    device = base_env.device
    num_envs = base_env.num_envs
    dt = float(base_env.step_dt)
    act_dim = resolve_action_dim(env)

    robot = base_env.scene[args_cli.robot_name]
    door = None
    try:
        door = base_env.scene[args_cli.door_name]
    except Exception:
        door = None

    robot_body_names = list(robot.data.body_names)
    hand_body_id = find_body_id(robot_body_names, args_cli.hand_body, None)
    left_body_id = find_body_id(robot_body_names, args_cli.left_body, args_cli.left_body_fallback)
    right_body_id = find_body_id(robot_body_names, args_cli.right_body, args_cli.right_body_fallback)

    robot_joint_names = list(robot.data.joint_names)
    if args_cli.gripper_joint not in robot_joint_names:
        raise RuntimeError(f"Gripper joint '{args_cli.gripper_joint}' not found in {robot_joint_names}")
    gripper_jid = robot_joint_names.index(args_cli.gripper_joint)

    try:
        left_sensor = base_env.scene[args_cli.left_contact_sensor]
    except Exception:
        left_sensor = None
    try:
        right_sensor = base_env.scene[args_cli.right_contact_sensor]
    except Exception:
        right_sensor = None

    open_axis_hand = safe_unit(parse_vec3(args_cli.open_axis_hand).to(device=device).unsqueeze(0))[0]

    open_steps = max(1, int(args_cli.open_seconds / max(1e-9, dt)))
    close_steps = max(1, int(args_cli.close_seconds / max(1e-9, dt)))
    post_rot_open_steps = max(1, int(args_cli.post_rotate_open_seconds / max(1e-9, dt)))
    rotate_max_steps = max(1, int(args_cli.rot_max_seconds / max(1e-9, dt)))

    phases = [
        Phase("open_air_1", "hold", args_cli.gripper_open_action, open_steps),
        Phase("close_air_1", "hold", args_cli.gripper_close_action, close_steps),
        Phase("open_air_2", "hold", args_cli.gripper_open_action, open_steps),
        Phase("rotate_90_open", "rotate", args_cli.gripper_open_action, rotate_max_steps),
        Phase("open_rot_1", "hold", args_cli.gripper_open_action, post_rot_open_steps),
        Phase("close_rot_1", "hold", args_cli.gripper_close_action, close_steps),
        Phase("open_rot_2", "hold", args_cli.gripper_open_action, open_steps),
    ]
    script_state = ScriptState(phases)

    obs, _ = env.reset()

    def compute_stats_dict():
        q_grip_all = robot.data.joint_pos[:, gripper_jid]
        width_drv_all = 2.0 * q_grip_all
        closed_drv_all = 1.0 - torch.clamp(width_drv_all / float(args_cli.stats_open_width), 0.0, 1.0)

        pL = robot.data.body_pos_w[:, left_body_id, :]
        pR = robot.data.body_pos_w[:, right_body_id, :]
        pHnd = robot.data.body_pos_w[:, hand_body_id, :]
        qHnd = robot.data.body_quat_w[:, hand_body_id, :]

        qHnd_c = quat_conjugate(qHnd)
        pL_h = quat_rotate(qHnd_c, pL - pHnd)
        pR_h = quat_rotate(qHnd_c, pR - pHnd)

        left_proj = torch.sum(pL_h * open_axis_hand.unsqueeze(0), dim=-1)
        right_proj = torch.sum(pR_h * open_axis_hand.unsqueeze(0), dim=-1)
        gap_proj_all = torch.abs(left_proj - right_proj)
        gap_euc_all = torch.linalg.norm(pL - pR, dim=-1)
        asym_all = torch.abs(torch.abs(left_proj) - torch.abs(right_proj))
        closed_geom_all = 1.0 - torch.clamp(gap_proj_all / float(args_cli.stats_open_width), 0.0, 1.0)

        fL_all = filtered_force_norm(left_sensor, num_envs, device)
        fR_all = filtered_force_norm(right_sensor, num_envs, device)

        return {
            "phase": script_state.phase.name if not script_state.is_finished else "finished",
            "phase_step": script_state.phase_step,
            "rot_deg": float(script_state.rotate_angle_deg),
            "rot_target_deg": float(args_cli.rot_target_deg),
            "q_env0": float(q_grip_all[0].item()),
            "width_drv_env0": float(width_drv_all[0].item()),
            "gap_proj_env0": float(gap_proj_all[0].item()),
            "gap_euc_env0": float(gap_euc_all[0].item()),
            "closed_drv_env0": float(closed_drv_all[0].item()),
            "closed_geom_env0": float(closed_geom_all[0].item()),
            "closed_mean_drv": float(closed_drv_all.mean().item()),
            "closed_mean_geom": float(closed_geom_all.mean().item()),
            "left_proj_env0": float(left_proj[0].item()),
            "right_proj_env0": float(right_proj[0].item()),
            "asym_env0": float(asym_all[0].item()),
            "fL_env0": float(fL_all[0].item()),
            "fR_env0": float(fR_all[0].item()),
        }

    hud = None
    if not args_cli.headless:
        hud = SymmetryHUD(compute_stats_dict, script_state, update_hz=args_cli.hud_update_hz)

    step_count = 0
    while simulation_app.is_running():
        start_t = time.time()
        if script_state.is_finished:
            # Hold last pose open after finishing.
            actions = torch.zeros((num_envs, act_dim), device=device)
            actions[:, -1] = args_cli.gripper_open_action
        else:
            phase = script_state.phase
            actions = torch.zeros((num_envs, act_dim), device=device)
            actions[:, -1] = float(phase.gripper_action)

            if phase.kind == "rotate":
                if args_cli.rot_action_dim < 0 or args_cli.rot_action_dim >= act_dim - 1:
                    raise RuntimeError(
                        f"rot_action_dim={args_cli.rot_action_dim} is invalid for action dim {act_dim}; "
                        "expected an arm dimension before the last gripper dim."
                    )
                actions[:, args_cli.rot_action_dim] = float(args_cli.rot_sign) * float(args_cli.rot_action_mag)

                q_cur = robot.data.body_quat_w[0:1, hand_body_id, :]
                if script_state.rotate_q_ref is None:
                    script_state.rotate_q_ref = q_cur.clone()
                script_state.rotate_angle_deg = quat_relative_angle_deg(script_state.rotate_q_ref, q_cur)
                if script_state.rotate_angle_deg >= float(args_cli.rot_target_deg):
                    script_state.rotate_done = True

        step_out = env.step(actions)
        if len(step_out) == 5:
            obs, rew, terminated, truncated, extras = step_out
            done = terminated | truncated
        else:
            obs, rew, done, extras = step_out

        if not script_state.is_finished:
            phase = script_state.phase
            script_state.phase_step += 1
            should_advance = False
            if phase.kind == "hold":
                should_advance = script_state.phase_step >= phase.duration_steps
            else:
                should_advance = script_state.rotate_done or (script_state.phase_step >= phase.duration_steps)
            if should_advance:
                script_state.advance()

        if args_cli.log_every > 0 and (step_count % args_cli.log_every == 0):
            s = compute_stats_dict()
            print(
                f"[SYMTEST] step={step_count} phase={s['phase']} rot={s['rot_deg']:.1f}/{s['rot_target_deg']:.1f}deg "
                f"q={s['q_env0']:.4f} width_drv={s['width_drv_env0']:.4f} gap_proj={s['gap_proj_env0']:.4f} "
                f"gap_euc={s['gap_euc_env0']:.4f} asym={s['asym_env0']:.4f} "
                f"closed_drv={s['closed_drv_env0']:.3f} closed_geom={s['closed_geom_env0']:.3f} "
                f"|F|L/R={s['fL_env0']:.2f}/{s['fR_env0']:.2f}"
            )

        step_count += 1

        if args_cli.real_time:
            sleep_t = dt - (time.time() - start_t)
            if sleep_t > 0:
                time.sleep(sleep_t)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
