# joint_pos_min = [-3.14, -0.05, -0.10, -1.60, -1.57, -2.00]
# joint_pos_max = [ 2.618, 3.50,  3.20,  1.55,  1.57,  2.00]
# joint_vel_max = [5.0, 5.0, 5.5, 5.5, 5.0, 5.0]
# joint_torque_max = [30.0, 40.0, 30.0, 15.0, 10.0, 10.0]
# default_kp = [80.0, 70.0, 70.0, 30.0, 30.0, 20.0]
# default_kd = [2.0, 2.0, 2.0, 1.0, 1.0, 0.7]
# controller_dt = 0.002 

from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


def _as_tensor(x, device, dim: int, name: str) -> torch.Tensor:
    t = torch.tensor(x, dtype=torch.float32, device=device).flatten()
    if t.numel() == 1:
        t = t.repeat(dim)
    if t.numel() != dim:
        raise ValueError(f"{name} must have length 1 or {dim}, got {t.numel()}")
    return t


class PiperHookMITJointAction(ActionTerm):
    """MIT-style joint impedance action for the Piper hook 6-DOF arm.

    Policy action -> q_des
    Per sim step -> tau = Kp * (q_des - q) + Kd * (dq_des - dq) + tau_ff
    """

    cfg: "PiperHookMITJointActionCfg"

    def __init__(self, cfg: "PiperHookMITJointActionCfg", env):
        super().__init__(cfg, env)

        self._env = env
        self._asset: Articulation = env.scene[cfg.asset_name]

        # resolve joints
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=True)
        self._num_joints = len(self._joint_ids)

        # buffers
        self._raw_actions = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._q_des = torch.zeros_like(self._raw_actions)
        self._dq_des = torch.zeros_like(self._raw_actions)
        self._tau_ff = torch.zeros_like(self._raw_actions)
        self._tau_cmd = torch.zeros_like(self._raw_actions)
        self._applied_delta = torch.zeros_like(self._raw_actions)
        self._prev_unlock_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_q_des_sync = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # params
        self._kp = _as_tensor(cfg.kp, self.device, self._num_joints, "kp").view(1, -1)
        self._kd = _as_tensor(cfg.kd, self.device, self._num_joints, "kd").view(1, -1)
        self._q_nominal = _as_tensor(cfg.nominal_joint_pos, self.device, self._num_joints, "nominal_joint_pos").view(1, -1)
        self._q_min = _as_tensor(cfg.joint_pos_min, self.device, self._num_joints, "joint_pos_min").view(1, -1)
        self._q_max = _as_tensor(cfg.joint_pos_max, self.device, self._num_joints, "joint_pos_max").view(1, -1)
        self._tau_limit = _as_tensor(cfg.effort_limit, self.device, self._num_joints, "effort_limit").view(1, -1)
        self._dq_limit = _as_tensor(cfg.velocity_limit, self.device, self._num_joints, "velocity_limit").view(1, -1)
        self._delta_scale = _as_tensor(cfg.delta_scale, self.device, self._num_joints, "delta_scale").view(1, -1)
        self._armature = _as_tensor(cfg.armature, self.device, self._num_joints, "armature").view(1, -1)
        self._stage2_tracking_error_limit = _as_tensor(
            cfg.stage2_tracking_error_limit,
            self.device,
            self._num_joints,
            "stage2_tracking_error_limit",
        ).view(1, -1)

        # init desired posture = nominal
        self._q_des[:] = self._q_nominal
        self._dq_des.zero_()
        self._tau_ff.zero_()

        # important: remove implicit PD on arm joints in sim
        # otherwise your MIT torque will fight a hidden position controller
        self._asset.write_joint_stiffness_to_sim(0.0, joint_ids=self._joint_ids)
        self._asset.write_joint_damping_to_sim(0.0, joint_ids=self._joint_ids)

        # set joint limits / torque limits / velocity limits / armature in sim
        tau_lim_full = self._tau_limit.repeat(self.num_envs, 1)
        dq_lim_full = self._dq_limit.repeat(self.num_envs, 1)
        armature_full = self._armature.repeat(self.num_envs, 1)

        self._asset.write_joint_effort_limit_to_sim(tau_lim_full, joint_ids=self._joint_ids)
        self._asset.write_joint_velocity_limit_to_sim(dq_lim_full, joint_ids=self._joint_ids)
        self._asset.write_joint_armature_to_sim(armature_full, joint_ids=self._joint_ids)

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def joint_ids(self):
        return self._joint_ids

    @property
    def q_des(self) -> torch.Tensor:
        return self._q_des

    @property
    def tau_cmd(self) -> torch.Tensor:
        return self._tau_cmd

    @property
    def applied_delta(self) -> torch.Tensor:
        return self._applied_delta

    def _ensure_log(self) -> dict:
        if not hasattr(self._env, "extras") or self._env.extras is None:
            self._env.extras = {}
        if "log" not in self._env.extras:
            self._env.extras["log"] = {}
        return self._env.extras["log"]

    def _physical_door_unlocked(self) -> torch.Tensor:
        # Privileged Teacher training may use the true physical unlock state here.
        # For deployment or Student distillation, replace this gate with an
        # estimated_unlocked signal or a conservative control rule that does not
        # depend on privileged simulator state.
        if hasattr(self._env, "_door_lock_mode"):
            return self._env._door_lock_mode == 2
        if hasattr(self._env, "_door_unlocked"):
            return self._env._door_unlocked.to(dtype=torch.bool)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _apply_stage2_action_scale(self, arm_delta: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_stage2_action_scale:
            return arm_delta

        unlock_mask = self._physical_door_unlocked()
        if torch.any(unlock_mask):
            arm_delta[unlock_mask, : min(3, self._num_joints)] *= float(self.cfg.stage2_arm_scale)
            if self._num_joints > 3:
                arm_delta[unlock_mask, 3 : min(6, self._num_joints)] *= float(self.cfg.stage2_wrist_scale)

        return arm_delta

    def _log_delta_stats(
        self,
        raw_delta: torch.Tensor,
        scaled_delta: torch.Tensor,
        unlock_mask: torch.Tensor,
    ) -> None:
        log = self._ensure_log()
        zero = torch.zeros((), device=self.device)
        arm_slice = slice(0, min(3, self._num_joints))
        wrist_slice = slice(3, min(6, self._num_joints))

        log["stage2_action/active_ratio"] = unlock_mask.float().mean().detach()
        log["stage2_action/raw_delta_arm_abs_mean"] = raw_delta[:, arm_slice].abs().mean().detach()
        log["stage2_action/scaled_delta_arm_abs_mean"] = scaled_delta[:, arm_slice].abs().mean().detach()
        if self._num_joints > 3:
            log["stage2_action/raw_delta_wrist_abs_mean"] = raw_delta[:, wrist_slice].abs().mean().detach()
            log["stage2_action/scaled_delta_wrist_abs_mean"] = scaled_delta[:, wrist_slice].abs().mean().detach()
        else:
            log["stage2_action/raw_delta_wrist_abs_mean"] = zero
            log["stage2_action/scaled_delta_wrist_abs_mean"] = zero

        if torch.any(unlock_mask):
            log["stage2_action/stage2_scaled_delta_arm_abs_mean"] = (
                scaled_delta[unlock_mask, arm_slice].abs().mean().detach()
            )
            if self._num_joints > 3:
                log["stage2_action/stage2_scaled_delta_wrist_abs_mean"] = (
                    scaled_delta[unlock_mask, wrist_slice].abs().mean().detach()
                )
            else:
                log["stage2_action/stage2_scaled_delta_wrist_abs_mean"] = zero
        else:
            log["stage2_action/stage2_scaled_delta_arm_abs_mean"] = zero
            log["stage2_action/stage2_scaled_delta_wrist_abs_mean"] = zero

    def _sync_pending_targets_if_needed(self) -> None:
        pending = self._pending_q_des_sync.clone()
        if torch.any(pending):
            q = self._asset.data.joint_pos[:, self._joint_ids]
            self._q_des[pending] = q[pending]
            self._dq_des[pending] = 0.0
            self._tau_ff[pending] = 0.0
            self._applied_delta[pending] = 0.0
            self._pending_q_des_sync[pending] = False

        log = self._ensure_log()
        log["mit_control/pending_sync_ratio"] = pending.float().mean().detach()

    def _sync_targets_on_unlock_transition_if_needed(self) -> None:
        unlock_mask = self._physical_door_unlocked()
        newly_unlocked = unlock_mask & (~self._prev_unlock_mask)

        if torch.any(newly_unlocked):
            q = self._asset.data.joint_pos[:, self._joint_ids]
            self._q_des[newly_unlocked] = q[newly_unlocked]
            self._dq_des[newly_unlocked] = 0.0
            self._tau_ff[newly_unlocked] = 0.0
            self._applied_delta[newly_unlocked] = 0.0

        self._prev_unlock_mask[:] = unlock_mask

        log = self._ensure_log()
        log["stage2_action/newly_unlocked_sync_ratio"] = newly_unlocked.float().mean().detach()

    def _limit_stage2_tracking_error_if_needed(self) -> None:
        if not self.cfg.use_stage2_tracking_error_limit:
            return

        unlock_mask = self._physical_door_unlocked()
        if not torch.any(unlock_mask):
            return

        q = self._asset.data.joint_pos[:, self._joint_ids]
        low = q - self._stage2_tracking_error_limit
        high = q + self._stage2_tracking_error_limit
        self._q_des[unlock_mask] = torch.max(
            torch.min(self._q_des[unlock_mask], high[unlock_mask]),
            low[unlock_mask],
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._applied_delta[env_ids] = 0.0
        self._q_des[env_ids] = self._q_nominal
        self._dq_des[env_ids] = 0.0
        self._tau_ff[env_ids] = 0.0
        self._tau_cmd[env_ids] = 0.0
        self._prev_unlock_mask[env_ids] = False
        self._pending_q_des_sync[env_ids] = True

    def process_actions(self, actions: torch.Tensor):
        # policy action range is assumed to be roughly [-1, 1]
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)

        self._sync_pending_targets_if_needed()
        self._sync_targets_on_unlock_transition_if_needed()

        raw_delta = self._processed_actions * self._delta_scale
        arm_delta = raw_delta.clone()
        arm_delta = self._apply_stage2_action_scale(arm_delta)
        self._applied_delta[:] = arm_delta
        unlock_mask = self._physical_door_unlocked()
        self._log_delta_stats(raw_delta, arm_delta, unlock_mask)

        if self.cfg.use_delta_mode:
            # incremental desired joint update per env step
            self._q_des[:] = self._q_des + arm_delta
        else:
            # absolute desired joint posture around nominal
            self._q_des[:] = self._q_nominal + arm_delta

        self._q_des[:] = torch.clamp(self._q_des, self._q_min, self._q_max)
        self._limit_stage2_tracking_error_if_needed()

    def apply_actions(self):
        q = self._asset.data.joint_pos[:, self._joint_ids]
        dq = self._asset.data.joint_vel[:, self._joint_ids]

        tau_unclipped = self._kp * (self._q_des - q) + self._kd * (self._dq_des - dq) + self._tau_ff
        tau = torch.clamp(tau_unclipped, -self._tau_limit, self._tau_limit)

        self._tau_cmd[:] = tau
        q_des_error = self._q_des - q
        saturated = torch.abs(tau_unclipped) >= (self._tau_limit - 1e-6)
        unlock_mask = self._physical_door_unlocked()
        zero = torch.zeros((), device=self.device)
        log = self._ensure_log()
        log["mit_control/q_des_error_abs_mean"] = q_des_error.abs().mean().detach()
        log["mit_control/tau_cmd_abs_mean"] = tau.abs().mean().detach()
        log["mit_control/torque_saturation_ratio"] = saturated.float().mean().detach()
        if torch.any(unlock_mask):
            log["stage2_action/stage2_q_des_error_abs_mean"] = q_des_error[unlock_mask].abs().mean().detach()
            log["stage2_action/stage2_tau_cmd_abs_mean"] = tau[unlock_mask].abs().mean().detach()
            log["stage2_action/stage2_torque_saturation_ratio"] = saturated[unlock_mask].float().mean().detach()
        else:
            log["stage2_action/stage2_q_des_error_abs_mean"] = zero
            log["stage2_action/stage2_tau_cmd_abs_mean"] = zero
            log["stage2_action/stage2_torque_saturation_ratio"] = zero

        # If your local Isaac Lab build exposes this method, this is the right call.
        # If not, only this line usually needs local API adjustment.
        self._asset.set_joint_effort_target(tau, joint_ids=self._joint_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        return

    def _debug_vis_callback(self, event):
        return


@configclass
class PiperHookMITJointActionCfg(ActionTermCfg):
    """Config for Piper hook MIT-style joint impedance action."""

    class_type: type[ActionTerm] = PiperHookMITJointAction

    asset_name: str = "robot"
    joint_names: list[str] = MISSING

    # if True: q_des <- q_des + action * delta_scale
    # if False: q_des <- q_nominal + action * delta_scale
    use_delta_mode: bool = True

    # policy action scale in joint space (rad per env step if delta mode)
    delta_scale: tuple[float, ...] | float = 0.02
    use_stage2_action_scale: bool = True
    stage2_arm_scale: float = 0.60
    stage2_wrist_scale: float = 0.35
    use_stage2_tracking_error_limit: bool = True
    stage2_tracking_error_limit: tuple[float, ...] | float = (0.15, 0.15, 0.12, 0.10, 0.10, 0.10)

    # controller params
    kp: tuple[float, ...] | float = (80.0, 70.0, 70.0, 30.0, 30.0, 20.0)
    kd: tuple[float, ...] | float = (2.0, 2.0, 2.0, 1.0, 1.0, 0.7)

    # nominal posture / hard limits
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.20, 0.20, 0.0, 0.0, 0.0)
    joint_pos_min: tuple[float, ...] = (-3.14, -0.05, -0.10, -1.60, -1.57, -2.00)
    joint_pos_max: tuple[float, ...] = ( 2.618, 3.50,  3.20,  1.55,  1.57,  2.00)

    # effort / velocity limits
    effort_limit: tuple[float, ...] = (30.0, 40.0, 30.0, 15.0, 10.0, 10.0)
    velocity_limit: tuple[float, ...] = (5.0, 5.0, 5.5, 5.5, 5.0, 5.0)

    # sim stabilization
    armature: tuple[float, ...] | float = 0.01
