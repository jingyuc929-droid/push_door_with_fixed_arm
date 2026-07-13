"""Hierarchical action term for door opening with frozen quadruped locomotion."""

from __future__ import annotations

import os
import sys
import importlib.util
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .mit_actions import _as_tensor


LOW_LEVEL_POLICY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "low_level_locomotion", "model_30000.pt")
)
DEFAULT_RL_ALGORITHMS_SOURCE_ROOT = (
    "/home/jing/Downloads/rl_sim_env-dev_yzc-653bdb9f039139e507b7d52a405b6dfd0bb53be4/source/rl_sim_env"
)

LOW_LEVEL_ACTOR_CFG = {
    "actor": {
        "num_actor_obs": 70,
        "num_actions": 12,
        "actor_hidden_dims": [512, 256, 128],
        "actor_obs_normalization": False,
    },
    "privileged_encoder": {
        "num_privileged_obs": 43,
        "privileged_encoder_hidden_dims": [64, 32],
        "num_privileged_encoder_out": 16,
    },
    "heightmap_encoder": {
        "num_heightmap_obs": 187,
        "heightmap_encoder_hidden_dims": [64, 32],
        "num_heightmap_encoder_out": 32,
    },
    "critic": {
        "num_critic_obs": 99,
        "critic_hidden_dims": [512, 256, 128],
        "critic_obs_normalization": False,
    },
    "init_noise_std": 1.0,
    "noise_std_type": "scalar",
    "activation": "elu",
    "min_normalized_std": [0.01] * 12,
    "use_estimator_out": True,
}

LOW_LEVEL_VAE_CFG = {
    "encoder_in_dim": 235,
    "encoder_hidden_dims": [128],
    "encoder_out_dim": 64,
    "encoder_head_dim_dict": {"obs_vel": 3, "obs_com": 3, "obs_mass": 1, "obs_latent": 16},
    "decoder_in_dim": 23,
    "decoder_hidden_dims": [64, 128],
    "decoder_out_dim": 51,
    "activation": "elu",
}


def _call_policy(policy, obs: torch.Tensor):
    action = policy(obs)
    if isinstance(action, tuple):
        action = action[0]
    elif isinstance(action, dict):
        for key in ("actions", "action", "mean", "mu"):
            if key in action:
                action = action[key]
                break
        else:
            raise RuntimeError("Unsupported low-level policy dict output; expected an action-like key.")
    return action


class _OnnxLowLevelPolicy:
    def __init__(self, policy_path: str, device: str, history_length: int, obs_dim: int, use_cuda: bool):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        providers = [provider for provider in providers if provider in available] or ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(policy_path, providers=providers)
        self._device = device
        self._input_names = [input_info.name for input_info in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        self._history_length = int(history_length)
        self._obs_dim = int(obs_dim)

    def __call__(self, obs: torch.Tensor, obs_history: torch.Tensor | None = None) -> torch.Tensor:
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        feed = {"actor_obs" if "actor_obs" in self._input_names else self._input_names[0]: obs_np}
        if "vae_obs" in self._input_names:
            if obs_history is None:
                vae_obs = np.tile(obs_np, (1, self._history_length))
            else:
                vae_obs = obs_history.detach().cpu().numpy().astype(np.float32).reshape(obs_np.shape[0], -1)
            expected = self._history_length * self._obs_dim
            if vae_obs.shape[-1] != expected:
                raise RuntimeError(f"Low-level ONNX vae_obs has {vae_obs.shape[-1]} dims, expected {expected}.")
            feed["vae_obs"] = vae_obs
        action = self._session.run([self._output_name], feed)[0]
        return torch.as_tensor(action, device=self._device, dtype=obs.dtype)


class _CheckpointLowLevelPolicy(torch.nn.Module):
    """PyTorch checkpoint loader for the locomotion ActorCriticEncoder+VAEBlind policy."""

    def __init__(
        self,
        policy_path: str,
        device: str,
        actor_cfg: dict,
        vae_cfg: dict,
        rl_source_root: str | None = None,
    ):
        super().__init__()
        try:
            from tensordict import TensorDict
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Checkpoint locomotion policy requires tensordict in the active Isaac Python environment."
            ) from exc

        rl_source_root = rl_source_root or DEFAULT_RL_ALGORITHMS_SOURCE_ROOT
        _ensure_path_on_sys_path(rl_source_root)
        ActorCriticEncoder, VAEBlind = _load_low_level_policy_classes(rl_source_root)
        checkpoint = _safe_load_low_level_checkpoint(policy_path, device)

        self._tensor_dict_type = TensorDict
        self.actor = ActorCriticEncoder(actor_cfg).to(device)
        self.vae = VAEBlind(vae_cfg).to(device)
        self.actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.vae.load_state_dict(checkpoint["vae_state_dict"], strict=True)
        self.actor.eval()
        self.vae.eval()

    def forward(self, obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        vae_obs = obs_history.reshape(obs.shape[0], -1)
        with torch.inference_mode():
            code = self.vae.act_inference(vae_obs)
            full_obs = self._tensor_dict_type(
                {"actor_obs": obs, "estimator_out": code},
                batch_size=obs.shape[:-1],
            )
            return self.actor.act_inference(full_obs)


def _ensure_path_on_sys_path(path: str | None) -> None:
    if path and os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


def _load_module_from_file(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module spec from {file_path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_low_level_policy_classes(rl_source_root: str):
    """Load only the modules needed for inference, avoiding rl_algorithms.rsl_rl.modules.__init__."""
    modules_dir = os.path.join(rl_source_root, "rl_algorithms", "rsl_rl", "modules")
    actor_path = os.path.join(modules_dir, "actor_critic_locomotion.py")
    vae_path = os.path.join(modules_dir, "vae_blind.py")
    if not os.path.isfile(actor_path) or not os.path.isfile(vae_path):
        raise ModuleNotFoundError(
            "Could not find low-level ActorCriticEncoder/VAEBlind source files. "
            f"Expected them under: {modules_dir}"
        )
    actor_module = _load_module_from_file("_door_low_actor_critic_locomotion", actor_path)
    vae_module = _load_module_from_file("_door_low_vae_blind", vae_path)
    return actor_module.ActorCriticEncoder, vae_module.VAEBlind


def _safe_load_low_level_checkpoint(policy_path: str, device: str):
    """Load numpy-2 pickled checkpoints in the current numpy-1 Isaac env."""
    import numpy as _np
    import numpy.core.multiarray as _np_multiarray
    from torch.serialization import safe_globals
    from rl_algorithms.amp_utils.normalizer import Normalizer

    safe_items = [
        Normalizer,
        _np.ndarray,
        _np.dtype,
        type(_np.dtype(_np.float64)),
        (_np_multiarray._reconstruct, "numpy._core.multiarray._reconstruct"),
    ]
    with safe_globals(safe_items):
        return torch.load(policy_path, map_location=device, weights_only=True)


class HighLevelDoorOpenAction(ActionTerm):
    """Action split: base command for frozen locomotion + arm joint position targets."""

    cfg: "HighLevelDoorOpenActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "HighLevelDoorOpenActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._env = env
        self._asset: Articulation = env.scene[cfg.asset_name]

        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            cfg.leg_joint_names, preserve_order=cfg.preserve_order
        )
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.arm_joint_names, preserve_order=cfg.preserve_order
        )
        self._num_base_actions = 5
        self._num_leg_actions = len(self._leg_joint_ids)
        self._num_joints = len(self._joint_ids)

        self._raw_actions = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._prev_high_action = torch.zeros_like(self._raw_actions)
        self._prev_prev_high_action = torch.zeros_like(self._raw_actions)

        self._q_des = torch.zeros((self.num_envs, self._num_joints), device=self.device)
        self._dq_des = torch.zeros_like(self._q_des)
        self._tau_ff = torch.zeros_like(self._q_des)
        self._tau_cmd = torch.zeros_like(self._q_des)
        self._applied_delta = torch.zeros_like(self._q_des)
        self._prev_unlock_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pending_q_des_sync = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._q_nominal = _as_tensor(cfg.nominal_joint_pos, self.device, self._num_joints, "nominal_joint_pos").view(1, -1)
        self._q_min = _as_tensor(cfg.joint_pos_min, self.device, self._num_joints, "joint_pos_min").view(1, -1)
        self._q_max = _as_tensor(cfg.joint_pos_max, self.device, self._num_joints, "joint_pos_max").view(1, -1)
        self._tau_limit = _as_tensor(cfg.effort_limit, self.device, self._num_joints, "effort_limit").view(1, -1)
        self._dq_limit = _as_tensor(cfg.velocity_limit, self.device, self._num_joints, "velocity_limit").view(1, -1)
        self._arm_scale = _as_tensor(cfg.arm_action_scale, self.device, self._num_joints, "arm_action_scale").view(1, -1)
        self._armature = _as_tensor(cfg.armature, self.device, self._num_joints, "armature").view(1, -1)
        self._stage2_tracking_error_limit = _as_tensor(
            cfg.stage2_tracking_error_limit,
            self.device,
            self._num_joints,
            "stage2_tracking_error_limit",
        ).view(1, -1)
        self._q_des[:] = self._q_nominal

        arm_stiffness = _as_tensor(cfg.arm_stiffness, self.device, self._num_joints, "arm_stiffness").view(1, -1)
        arm_damping = _as_tensor(cfg.arm_damping, self.device, self._num_joints, "arm_damping").view(1, -1)
        self._asset.write_joint_stiffness_to_sim(arm_stiffness.repeat(self.num_envs, 1), joint_ids=self._joint_ids)
        self._asset.write_joint_damping_to_sim(arm_damping.repeat(self.num_envs, 1), joint_ids=self._joint_ids)
        self._asset.write_joint_effort_limit_to_sim(self._tau_limit.repeat(self.num_envs, 1), joint_ids=self._joint_ids)
        self._asset.write_joint_velocity_limit_to_sim(self._dq_limit.repeat(self.num_envs, 1), joint_ids=self._joint_ids)
        self._asset.write_joint_armature_to_sim(self._armature.repeat(self.num_envs, 1), joint_ids=self._joint_ids)

        self._low_level_obs_dim = int(cfg.low_level_obs_dim)
        self._low_level_history_length = int(cfg.low_level_history_length)
        self._low_level_action_scale = float(cfg.low_level_action_scale)
        self._low_level_decimation = int(cfg.low_level_decimation)
        self._low_level_base_ang_vel_scale = float(cfg.low_level_base_ang_vel_scale)
        self._low_level_joint_vel_scale = float(cfg.low_level_joint_vel_scale)
        self._low_level_last_action_scale = float(cfg.low_level_last_action_scale)
        self._low_level_command_scale = tuple(cfg.low_level_command_scale)
        self._base_command = torch.zeros(self.num_envs, 5, device=self.device)
        self._requested_base_command = torch.zeros_like(self._base_command)
        self._prev_base_command = torch.zeros_like(self._base_command)
        self._prev_prev_base_command = torch.zeros_like(self._base_command)
        self._base_low = torch.tensor(cfg.base_command_low, device=self.device, dtype=torch.float32).view(1, 5)
        self._base_high = torch.tensor(cfg.base_command_high, device=self.device, dtype=torch.float32).view(1, 5)
        self._base_mid = 0.5 * (self._base_low + self._base_high)
        self._base_half_range = 0.5 * (self._base_high - self._base_low)
        self._last_low_action = torch.zeros(self.num_envs, self._num_leg_actions, device=self.device)
        self._low_obs_history = torch.zeros(
            self.num_envs,
            self._low_level_history_length,
            self._low_level_obs_dim,
            device=self.device,
        )
        self._leg_target = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._sim_counter_since_low_update = 0

        self._low_policy = None
        self._low_policy_kind = "none"
        if cfg.low_level_policy_path:
            self._load_low_policy(cfg)

        self._publish_buffers()

    @property
    def action_dim(self) -> int:
        return self._num_base_actions + self._num_joints

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

    @property
    def base_command(self) -> torch.Tensor:
        return self._base_command

    def _load_low_policy(self, cfg: "HighLevelDoorOpenActionCfg") -> None:
        policy_format = cfg.low_level_policy_format
        if policy_format == "auto":
            if cfg.low_level_policy_path.endswith(".onnx"):
                policy_format = "onnx"
            elif cfg.low_level_policy_path.endswith(".pt"):
                policy_format = "checkpoint"
            else:
                policy_format = "torchscript"
        if policy_format == "onnx":
            self._low_policy = _OnnxLowLevelPolicy(
                cfg.low_level_policy_path,
                self.device,
                self._low_level_history_length,
                self._low_level_obs_dim,
                cfg.low_level_onnx_use_cuda,
            )
            self._low_policy_kind = "onnx"
        elif policy_format == "checkpoint":
            self._low_policy = _CheckpointLowLevelPolicy(
                cfg.low_level_policy_path,
                self.device,
                cfg.low_level_actor_cfg or LOW_LEVEL_ACTOR_CFG,
                cfg.low_level_vae_cfg or LOW_LEVEL_VAE_CFG,
                cfg.low_level_rl_algorithms_source_root,
            )
            self._low_policy_kind = "checkpoint"
        else:
            self._low_policy = torch.jit.load(cfg.low_level_policy_path, map_location=self.device)
            self._low_policy.eval()
            self._low_policy_kind = "torchscript"

    def _publish_buffers(self) -> None:
        setattr(self._env, "high_level_base_command", self._base_command)
        setattr(self._env, "high_level_requested_base_command", self._requested_base_command)
        setattr(self._env, "high_level_previous_base_command", self._prev_base_command)
        setattr(self._env, "high_level_previous_previous_base_command", self._prev_prev_base_command)
        setattr(self._env, "high_level_previous_action", self._prev_high_action)
        setattr(self._env, "high_level_previous_previous_action", self._prev_prev_high_action)
        setattr(self._env, "high_level_arm_target", self._q_des)
        setattr(self._env, "low_level_last_action", self._last_low_action)

    def _ensure_log(self) -> dict:
        if not hasattr(self._env, "extras") or self._env.extras is None:
            self._env.extras = {}
        if "log" not in self._env.extras:
            self._env.extras["log"] = {}
        return self._env.extras["log"]

    def _physical_door_unlocked(self) -> torch.Tensor:
        if hasattr(self._env, "_door_lock_mode"):
            return self._env._door_lock_mode == 2
        if hasattr(self._env, "_door_unlocked"):
            return self._env._door_unlocked.to(dtype=torch.bool)
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _sync_pending_targets_if_needed(self) -> None:
        pending = self._pending_q_des_sync.clone()
        if torch.any(pending):
            q = self._asset.data.joint_pos[:, self._joint_ids]
            self._q_des[pending] = q[pending]
            self._dq_des[pending] = 0.0
            self._tau_ff[pending] = 0.0
            self._applied_delta[pending] = 0.0
            self._pending_q_des_sync[pending] = False
        self._ensure_log()["arm_position_control/pending_sync_ratio"] = pending.float().mean().detach()

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
        self._ensure_log()["stage2_action/newly_unlocked_sync_ratio"] = newly_unlocked.float().mean().detach()

    def _apply_stage2_action_scale(self, arm_delta: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_stage2_action_scale:
            return arm_delta
        unlock_mask = self._physical_door_unlocked()
        if torch.any(unlock_mask):
            arm_delta[unlock_mask, : min(3, self._num_joints)] *= float(self.cfg.stage2_arm_scale)
            if self._num_joints > 3:
                arm_delta[unlock_mask, 3 : min(6, self._num_joints)] *= float(self.cfg.stage2_wrist_scale)
        return arm_delta

    def _limit_stage2_tracking_error_if_needed(self) -> None:
        if not self.cfg.use_stage2_tracking_error_limit:
            return
        unlock_mask = self._physical_door_unlocked()
        if not torch.any(unlock_mask):
            return
        q = self._asset.data.joint_pos[:, self._joint_ids]
        low = q - self._stage2_tracking_error_limit
        high = q + self._stage2_tracking_error_limit
        self._q_des[unlock_mask] = torch.max(torch.min(self._q_des[unlock_mask], high[unlock_mask]), low[unlock_mask])

    def _log_delta_stats(self, raw_delta: torch.Tensor, scaled_delta: torch.Tensor, unlock_mask: torch.Tensor) -> None:
        log = self._ensure_log()
        zero = torch.zeros((), device=self.device)
        arm_slice = slice(0, min(3, self._num_joints))
        wrist_slice = slice(3, min(6, self._num_joints))
        log["stage2_action/active_ratio"] = unlock_mask.float().mean().detach()
        log["arm_position_action/raw_target_delta_arm_abs_mean"] = raw_delta[:, arm_slice].abs().mean().detach()
        log["arm_position_action/applied_target_delta_arm_abs_mean"] = scaled_delta[:, arm_slice].abs().mean().detach()
        if self._num_joints > 3:
            log["arm_position_action/raw_target_delta_wrist_abs_mean"] = raw_delta[:, wrist_slice].abs().mean().detach()
            log["arm_position_action/applied_target_delta_wrist_abs_mean"] = scaled_delta[:, wrist_slice].abs().mean().detach()
        else:
            log["arm_position_action/raw_target_delta_wrist_abs_mean"] = zero
            log["arm_position_action/applied_target_delta_wrist_abs_mean"] = zero

    def process_actions(self, actions: torch.Tensor):
        self._prev_prev_high_action[:] = self._prev_high_action
        self._prev_high_action[:] = self._raw_actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, -1.0, 1.0)

        base_action = self._processed_actions[:, :5]
        target_base_command = self._base_mid + base_action * self._base_half_range
        self._requested_base_command[:] = target_base_command
        target_base_command = target_base_command.clone()
        target_base_command[:, 3] = float(self.cfg.default_body_height)
        target_base_command[:, 4] = 0.0
        alpha = max(0.0, min(1.0, float(self.cfg.command_smoothing_alpha)))
        self._prev_prev_base_command[:] = self._prev_base_command
        self._prev_base_command[:] = self._base_command
        self._base_command[:] = alpha * target_base_command + (1.0 - alpha) * self._base_command

        self._sync_pending_targets_if_needed()
        self._sync_targets_on_unlock_transition_if_needed()
        arm_action = self._processed_actions[:, 5:]
        if self.cfg.use_default_arm_offset:
            arm_offset = self._asset.data.default_joint_pos[:, self._joint_ids]
        else:
            arm_offset = self._q_nominal
        desired_arm_target = arm_offset + arm_action * self._arm_scale
        desired_arm_target = torch.clamp(desired_arm_target, self._q_min, self._q_max)

        arm_alpha = max(0.0, min(1.0, float(self.cfg.arm_target_smoothing_alpha)))
        if arm_alpha < 1.0:
            desired_arm_target = arm_alpha * desired_arm_target + (1.0 - arm_alpha) * self._q_des

        raw_delta = desired_arm_target - self._q_des
        max_delta = float(self.cfg.arm_target_max_delta) if self.cfg.arm_target_max_delta is not None else 0.0
        if max_delta > 0.0:
            raw_delta = torch.clamp(raw_delta, min=-max_delta, max=max_delta)
        arm_delta = self._apply_stage2_action_scale(raw_delta.clone())
        self._applied_delta[:] = arm_delta
        self._log_delta_stats(raw_delta, arm_delta, self._physical_door_unlocked())

        self._q_des[:] = self._q_des + arm_delta
        self._q_des[:] = torch.clamp(self._q_des, self._q_min, self._q_max)
        self._limit_stage2_tracking_error_if_needed()

        self._update_low_level_target(force=True)
        self._publish_buffers()

    def apply_actions(self):
        if self._sim_counter_since_low_update >= max(self._low_level_decimation, 1):
            self._update_low_level_target(force=True)
        self._sim_counter_since_low_update += 1
        self._asset.set_joint_position_target(self._leg_target, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_position_target(self._q_des, joint_ids=self._joint_ids)

        q_des_error = self._q_des - self._asset.data.joint_pos[:, self._joint_ids]
        log = self._ensure_log()
        log["arm_position_control/q_des_error_abs_mean"] = q_des_error.abs().mean().detach()

    def _build_low_level_obs(self) -> torch.Tensor:
        joint_pos_rel = (
            self._asset.data.joint_pos[:, self._leg_joint_ids]
            - self._asset.data.default_joint_pos[:, self._leg_joint_ids]
        )
        joint_vel = self._asset.data.joint_vel[:, self._leg_joint_ids] * self._low_level_joint_vel_scale
        command_scale = torch.tensor(self._low_level_command_scale, device=self.device, dtype=torch.float32).view(1, 5)
        obs = torch.cat(
            (
                self._asset.data.root_ang_vel_b * self._low_level_base_ang_vel_scale,
                self._asset.data.projected_gravity_b,
                joint_pos_rel,
                joint_vel,
                self._last_low_action * self._low_level_last_action_scale,
                self._base_command * command_scale,
            ),
            dim=-1,
        )
        if obs.shape[-1] != self._low_level_obs_dim:
            raise RuntimeError(f"Low-level obs has {obs.shape[-1]} dims, expected {self._low_level_obs_dim}.")
        return obs

    def _update_low_level_target(self, force: bool = False):
        if not force:
            return
        if self._low_policy is None:
            low_action = torch.zeros(self.num_envs, self._num_leg_actions, device=self.device)
        else:
            low_obs = self._build_low_level_obs()
            with torch.inference_mode():
                self._low_obs_history = torch.roll(self._low_obs_history, shifts=-1, dims=1)
                self._low_obs_history[:, -1, :] = low_obs
                if self._low_policy_kind in ("onnx", "checkpoint"):
                    low_action = self._low_policy(low_obs, self._low_obs_history)
                else:
                    low_action = _call_policy(self._low_policy, low_obs)
        if low_action.shape[-1] != self._num_leg_actions:
            raise RuntimeError(
                f"Low-level policy produced {low_action.shape[-1]} actions, expected {self._num_leg_actions}."
            )
        leg_default = self._asset.data.default_joint_pos[:, self._leg_joint_ids]
        self._leg_target[:] = leg_default + low_action * self._low_level_action_scale
        self._last_low_action[:] = low_action
        self._sim_counter_since_low_update = 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._prev_high_action[env_ids] = 0.0
        self._prev_prev_high_action[env_ids] = 0.0
        self._q_des[env_ids] = self._q_nominal
        self._dq_des[env_ids] = 0.0
        self._tau_ff[env_ids] = 0.0
        self._tau_cmd[env_ids] = 0.0
        self._applied_delta[env_ids] = 0.0
        self._prev_unlock_mask[env_ids] = False
        self._pending_q_des_sync[env_ids] = True
        self._last_low_action[env_ids] = 0.0
        self._requested_base_command[env_ids] = self._base_mid
        self._base_command[env_ids] = self._base_mid
        self._base_command[env_ids, 3] = float(self.cfg.default_body_height)
        self._base_command[env_ids, 4] = 0.0
        self._prev_base_command[env_ids] = self._base_mid
        self._prev_base_command[env_ids, 3] = float(self.cfg.default_body_height)
        self._prev_base_command[env_ids, 4] = 0.0
        self._prev_prev_base_command[env_ids] = self._base_mid
        self._prev_prev_base_command[env_ids, 3] = float(self.cfg.default_body_height)
        self._prev_prev_base_command[env_ids, 4] = 0.0
        default_pos = self._asset.data.default_joint_pos
        self._leg_target[env_ids] = default_pos[env_ids][:, self._leg_joint_ids]
        reset_obs = self._build_low_level_obs()
        self._low_obs_history[env_ids] = reset_obs[env_ids].unsqueeze(1).repeat(1, self._low_level_history_length, 1)
        self._sim_counter_since_low_update = self._low_level_decimation
        self._publish_buffers()

    def _set_debug_vis_impl(self, debug_vis: bool):
        return

    def _debug_vis_callback(self, event):
        return


@configclass
class HighLevelDoorOpenActionCfg(ActionTermCfg):
    """Teacher action: first 5 dims are base command, then 6 arm position-target dims."""

    class_type: type[ActionTerm] = HighLevelDoorOpenAction

    asset_name: str = "robot"
    leg_joint_names: list[str] = MISSING
    arm_joint_names: list[str] = MISSING
    base_command_low: tuple[float, float, float, float, float] = (-0.45, -0.25, -0.5, 0.38, -0.15)
    base_command_high: tuple[float, float, float, float, float] = (0.8, 0.25, 0.5, 0.47, 0.28)
    command_smoothing_alpha: float = 0.35
    default_body_height: float = 0.43

    low_level_policy_path: str | None = LOW_LEVEL_POLICY_PATH
    low_level_policy_format: str = "checkpoint"
    low_level_actor_cfg: dict | None = LOW_LEVEL_ACTOR_CFG
    low_level_vae_cfg: dict | None = LOW_LEVEL_VAE_CFG
    low_level_rl_algorithms_source_root: str | None = DEFAULT_RL_ALGORITHMS_SOURCE_ROOT
    low_level_obs_dim: int = 47
    low_level_history_length: int = 5
    low_level_action_scale: float = 0.25
    low_level_decimation: int = 8
    low_level_onnx_use_cuda: bool = True
    low_level_base_ang_vel_scale: float = 0.25
    low_level_joint_vel_scale: float = 0.05
    low_level_last_action_scale: float = 0.25
    low_level_command_scale: tuple[float, float, float, float, float] = (2.0, 2.0, 0.25, 2.0, 1.0)

    arm_action_scale: tuple[float, ...] | float = (0.8, 0.8, 0.8, 0.8, 0.6, 0.6)
    arm_target_smoothing_alpha: float = 0.35
    arm_target_max_delta: float | None = 0.08
    use_default_arm_offset: bool = True
    use_stage2_action_scale: bool = True
    stage2_arm_scale: float = 0.60
    stage2_wrist_scale: float = 0.35
    use_stage2_tracking_error_limit: bool = True
    stage2_tracking_error_limit: tuple[float, ...] | float = (0.15, 0.15, 0.12, 0.10, 0.10, 0.10)
    arm_stiffness: tuple[float, ...] | float = 25.0
    arm_damping: tuple[float, ...] | float = 1.0
    nominal_joint_pos: tuple[float, ...] = (0.0, 0.20, -0.20, 0.0, 0.0, 0.0)
    joint_pos_min: tuple[float, ...] = (-2.6179938, 0.0, -2.9670597, -2.2165681, -1.5620696, -2.0943951)
    joint_pos_max: tuple[float, ...] = (2.6179938, 3.1415926, 0.0, 2.2165681, 1.5620696, 2.0943951)
    effort_limit: tuple[float, ...] = (50.0, 50.0, 50.0, 50.0, 50.0, 50.0)
    velocity_limit: tuple[float, ...] = (5.0, 5.0, 5.5, 5.5, 5.0, 5.0)
    armature: tuple[float, ...] | float = (0.02, 0.02, 0.02, 0.01, 0.01, 0.01)
    preserve_order: bool = True
