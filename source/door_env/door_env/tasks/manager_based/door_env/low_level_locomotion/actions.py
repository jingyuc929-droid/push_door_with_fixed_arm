"""Hierarchical high-level actions for pick-and-place tasks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import numpy as np
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass


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
    def __init__(
        self,
        policy_path: str,
        device: str,
        history_length: int,
        obs_dim: int,
        use_cuda: bool,
    ):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
        providers = [provider for provider in providers if provider in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        self._session = ort.InferenceSession(policy_path, providers=providers)
        self._device = device
        self._input_names = [input_info.name for input_info in self._session.get_inputs()]
        self._output_name = self._session.get_outputs()[0].name
        self._history_length = int(history_length)
        self._obs_dim = int(obs_dim)

    def __call__(self, obs: torch.Tensor, obs_history: torch.Tensor | None = None) -> torch.Tensor:
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        feed = {}
        if "actor_obs" in self._input_names:
            feed["actor_obs"] = obs_np
        else:
            feed[self._input_names[0]] = obs_np
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
    """PyTorch checkpoint loader for the lower locomotion ActorCritic+VAE policy."""

    def __init__(
        self,
        policy_path: str,
        device: str,
        actor_cfg: dict,
        vae_cfg: dict,
    ):
        super().__init__()
        from tensordict import TensorDict
        from rl_algorithms.rsl_rl.modules import ActorCriticEncoder, VAEBlind

        self._tensor_dict_type = TensorDict
        self.actor = ActorCriticEncoder(actor_cfg).to(device)
        self.vae = VAEBlind(vae_cfg).to(device)
        checkpoint = torch.load(policy_path, map_location=device, weights_only=False)
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


class HighLevelPickPlaceAction(ActionTerm):
    """Action term for a frozen-locomotion, trainable-manipulation hierarchy."""

    cfg: HighLevelPickPlaceActionCfg
    _asset: Articulation

    def __init__(self, cfg: HighLevelPickPlaceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            cfg.leg_joint_names, preserve_order=cfg.preserve_order
        )
        self._arm_joint_ids, self._arm_joint_names = self._asset.find_joints(
            cfg.arm_joint_names, preserve_order=cfg.preserve_order
        )
        self._num_base_actions = 5
        self._num_arm_actions = len(self._arm_joint_ids)
        self._num_leg_actions = len(self._leg_joint_ids)
        self._gripper_joint_ids, self._gripper_joint_names = self._find_gripper_joints(cfg)
        self._gripper_mimic_joint_ids, self._gripper_mimic_joint_names = self._find_gripper_mimic_joints(cfg)
        self._num_gripper_actions = len(self._gripper_joint_ids)
        self._low_level_obs_dim = int(getattr(cfg, "low_level_obs_dim", 47))
        self._low_level_history_length = int(getattr(cfg, "low_level_history_length", 5))
        self._low_level_action_scale = float(getattr(cfg, "low_level_action_scale", 0.25))
        self._low_level_decimation = int(getattr(cfg, "low_level_decimation", 4))
        self._low_level_onnx_use_cuda = bool(getattr(cfg, "low_level_onnx_use_cuda", True))
        self._low_level_base_ang_vel_scale = float(getattr(cfg, "low_level_base_ang_vel_scale", 0.25))
        self._low_level_joint_vel_scale = float(getattr(cfg, "low_level_joint_vel_scale", 0.05))
        self._low_level_last_action_scale = float(getattr(cfg, "low_level_last_action_scale", 0.25))
        self._low_level_command_scale = tuple(
            getattr(cfg, "low_level_command_scale", (2.0, 2.0, 0.25, 2.0, 1.0))
        )
        self._low_level_policy_path = getattr(cfg, "low_level_policy_path", None)
        self._low_level_policy_format = getattr(cfg, "low_level_policy_format", "auto")
        self._low_level_obs_key = getattr(cfg, "low_level_obs_key", "low_actor_obs")
        self._build_low_level_obs_from_state = bool(getattr(cfg, "build_low_level_obs_from_state", True))
        self._zero_leg_action_without_policy = bool(getattr(cfg, "zero_leg_action_without_policy", True))

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._base_command = torch.zeros(self.num_envs, 5, device=self.device)
        self._prev_base_command = torch.zeros_like(self._base_command)
        self._prev_prev_base_command = torch.zeros_like(self._base_command)
        self._arm_target = self._asset.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._gripper_target = (
            self._asset.data.default_joint_pos[:, self._gripper_joint_ids].clone()
            if self._num_gripper_actions > 0
            else torch.zeros(self.num_envs, 0, device=self.device)
        )
        self._gripper_mimic_target = (
            self._asset.data.default_joint_pos[:, self._gripper_mimic_joint_ids].clone()
            if self._gripper_mimic_joint_ids
            else torch.zeros(self.num_envs, 0, device=self.device)
        )
        self._leg_target = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        self._prev_high_action = torch.zeros_like(self._raw_actions)
        self._prev_prev_high_action = torch.zeros_like(self._raw_actions)
        self._last_low_action = torch.zeros(self.num_envs, self._num_leg_actions, device=self.device)
        self._low_obs_history = torch.zeros(
            self.num_envs,
            self._low_level_history_length,
            self._low_level_obs_dim,
            device=self.device,
        )
        self._sim_counter_since_low_update = 0

        self._base_low = torch.tensor(cfg.base_command_low, device=self.device, dtype=torch.float32).view(1, 5)
        self._base_high = torch.tensor(cfg.base_command_high, device=self.device, dtype=torch.float32).view(1, 5)
        self._base_mid = 0.5 * (self._base_low + self._base_high)
        self._base_half_range = 0.5 * (self._base_high - self._base_low)

        if isinstance(cfg.arm_action_scale, tuple):
            self._arm_scale = torch.tensor(cfg.arm_action_scale, device=self.device, dtype=torch.float32).view(1, -1)
            if self._arm_scale.shape[1] != self._num_arm_actions:
                raise ValueError(
                    f"arm_action_scale has {self._arm_scale.shape[1]} entries, expected {self._num_arm_actions}."
                )
        else:
            self._arm_scale = float(cfg.arm_action_scale)

        self._low_policy = None
        self._low_policy_kind = "none"
        if self._low_level_policy_path:
            policy_format = self._low_level_policy_format
            if policy_format == "auto":
                if self._low_level_policy_path.endswith(".onnx"):
                    policy_format = "onnx"
                elif self._low_level_policy_path.endswith(".pt"):
                    policy_format = "checkpoint"
                else:
                    policy_format = "torchscript"
            if policy_format == "onnx":
                self._low_policy = _OnnxLowLevelPolicy(
                    self._low_level_policy_path,
                    self.device,
                    self._low_level_history_length,
                    self._low_level_obs_dim,
                    self._low_level_onnx_use_cuda,
                )
                self._low_policy_kind = "onnx"
            elif policy_format == "checkpoint":
                self._low_policy = _CheckpointLowLevelPolicy(
                    self._low_level_policy_path,
                    self.device,
                    getattr(cfg, "low_level_actor_cfg", None) or {},
                    getattr(cfg, "low_level_vae_cfg", None) or {},
                )
                self._low_policy_kind = "checkpoint"
            else:
                self._low_policy = torch.jit.load(self._low_level_policy_path, map_location=self.device)
                self._low_policy.eval()
                self._low_policy_kind = "torchscript"

        setattr(env, "high_level_base_command", self._base_command)
        setattr(env, "high_level_previous_base_command", self._prev_base_command)
        setattr(env, "high_level_previous_previous_base_command", self._prev_prev_base_command)
        setattr(env, "high_level_previous_action", self._prev_high_action)
        setattr(env, "high_level_previous_previous_action", self._prev_prev_high_action)
        setattr(env, "high_level_arm_target", self._arm_target)
        setattr(env, "high_level_gripper_target", self._gripper_target)
        setattr(env, "low_level_last_action", self._last_low_action)

    @property
    def action_dim(self) -> int:
        return self._num_base_actions + self._num_arm_actions + self._num_gripper_actions

    def _find_gripper_joints(self, cfg: HighLevelPickPlaceActionCfg):
        gripper_joint_names = list(getattr(cfg, "gripper_joint_names", []) or [])
        if not gripper_joint_names:
            return [], []
        ids, names = self._asset.find_joints(gripper_joint_names, preserve_order=cfg.preserve_order)
        return ids, names

    def _find_gripper_mimic_joints(self, cfg: HighLevelPickPlaceActionCfg):
        gripper_joint_names = list(getattr(cfg, "gripper_mimic_joint_names", []) or [])
        if not gripper_joint_names:
            return [], []
        ids, names = self._asset.find_joints(gripper_joint_names, preserve_order=cfg.preserve_order)
        return ids, names

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def base_command(self) -> torch.Tensor:
        return self._base_command

    def process_actions(self, actions: torch.Tensor):
        self._prev_prev_high_action[:] = self._prev_high_action
        self._prev_high_action[:] = self._raw_actions
        self._raw_actions[:] = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions[:] = self._raw_actions

        base_action = self._raw_actions[:, :5]
        target_base_command = self._base_mid + base_action * self._base_half_range
        alpha = float(self.cfg.command_smoothing_alpha)
        alpha = max(0.0, min(1.0, alpha))
        self._prev_prev_base_command[:] = self._prev_base_command
        self._prev_base_command[:] = self._base_command
        self._base_command[:] = alpha * target_base_command + (1.0 - alpha) * self._base_command

        arm_action = self._raw_actions[:, 5 : 5 + self._num_arm_actions]
        gripper_action = self._raw_actions[:, 5 + self._num_arm_actions :]
        if self.cfg.use_default_arm_offset:
            arm_offset = self._asset.data.default_joint_pos[:, self._arm_joint_ids]
        else:
            arm_offset = 0.0
        desired_arm_target = arm_offset + arm_action * self._arm_scale
        arm_alpha = float(getattr(self.cfg, "arm_target_smoothing_alpha", 1.0))
        arm_alpha = max(0.0, min(1.0, arm_alpha))
        if arm_alpha < 1.0:
            desired_arm_target = arm_alpha * desired_arm_target + (1.0 - arm_alpha) * self._arm_target
        arm_max_delta = getattr(self.cfg, "arm_target_max_delta", None)
        if arm_max_delta is not None and float(arm_max_delta) > 0.0:
            delta = torch.clamp(
                desired_arm_target - self._arm_target,
                min=-float(arm_max_delta),
                max=float(arm_max_delta),
            )
            desired_arm_target = self._arm_target + delta
        self._arm_target[:] = desired_arm_target
        if self._num_gripper_actions > 0:
            if bool(getattr(self.cfg, "gripper_binary_action", False)):
                closed_target = float(getattr(self.cfg, "gripper_closed_target", 0.0))
                open_target = float(getattr(self.cfg, "gripper_open_target", 0.1))
                threshold = float(getattr(self.cfg, "gripper_binary_threshold", 0.0))
                self._gripper_target = torch.where(
                    gripper_action < threshold,
                    torch.full_like(gripper_action, closed_target),
                    torch.full_like(gripper_action, open_target),
                )
            else:
                gripper_offset = self._asset.data.default_joint_pos[:, self._gripper_joint_ids]
                gripper_scale = float(getattr(self.cfg, "gripper_action_scale", 0.03))
                self._gripper_target = gripper_offset + gripper_action * gripper_scale
            gripper_range = getattr(self.cfg, "gripper_target_range", None)
            if gripper_range is not None:
                self._gripper_target = torch.clamp(
                    self._gripper_target,
                    min=float(gripper_range[0]),
                    max=float(gripper_range[1]),
                )
            if self._gripper_mimic_joint_ids:
                if self._num_gripper_actions != 1:
                    raise RuntimeError("gripper_mimic_joint_names currently expects exactly one gripper action joint.")
                multipliers = torch.tensor(
                    getattr(self.cfg, "gripper_mimic_multipliers", ()),
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, -1)
                offsets = torch.tensor(
                    getattr(self.cfg, "gripper_mimic_offsets", ()),
                    device=self.device,
                    dtype=torch.float32,
                ).view(1, -1)
                if multipliers.shape[1] != len(self._gripper_mimic_joint_ids):
                    raise RuntimeError(
                        "gripper_mimic_multipliers must match gripper_mimic_joint_names length."
                    )
                if offsets.shape[1] != len(self._gripper_mimic_joint_ids):
                    raise RuntimeError("gripper_mimic_offsets must match gripper_mimic_joint_names length.")
                self._gripper_mimic_target = self._gripper_target[:, :1] * multipliers + offsets

        self._update_low_level_target(force=True)

        setattr(self._env, "high_level_base_command", self._base_command)
        setattr(self._env, "high_level_previous_base_command", self._prev_base_command)
        setattr(self._env, "high_level_previous_previous_base_command", self._prev_prev_base_command)
        setattr(self._env, "high_level_previous_action", self._prev_high_action)
        setattr(self._env, "high_level_previous_previous_action", self._prev_prev_high_action)
        setattr(self._env, "high_level_arm_target", self._arm_target)
        setattr(self._env, "high_level_gripper_target", self._gripper_target)
        setattr(self._env, "low_level_last_action", self._last_low_action)

    def apply_actions(self):
        if self._sim_counter_since_low_update >= max(self._low_level_decimation, 1):
            self._update_low_level_target(force=True)
        self._sim_counter_since_low_update += 1
        self._asset.set_joint_position_target(self._leg_target, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_position_target(self._arm_target, joint_ids=self._arm_joint_ids)
        if self._num_gripper_actions > 0:
            self._asset.set_joint_position_target(self._gripper_target, joint_ids=self._gripper_joint_ids)
        if self._gripper_mimic_joint_ids:
            self._asset.set_joint_position_target(self._gripper_mimic_target, joint_ids=self._gripper_mimic_joint_ids)

    def _build_low_level_obs(self) -> torch.Tensor:
        joint_pos_rel = self._asset.data.joint_pos[:, self._leg_joint_ids] - self._asset.data.default_joint_pos[
            :, self._leg_joint_ids
        ]
        joint_vel = self._asset.data.joint_vel[:, self._leg_joint_ids] * self._low_level_joint_vel_scale
        command_scale = torch.tensor(
            self._low_level_command_scale,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 5)
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
            if not self._zero_leg_action_without_policy:
                low_action[:, : min(self.action_dim, self._num_leg_actions)] = self._raw_actions[
                    :, : min(self.action_dim, self._num_leg_actions)
                ]
        else:
            if self._build_low_level_obs_from_state:
                low_obs = self._build_low_level_obs()
            else:
                low_obs = getattr(self._env, "obs_tensor_dict", {}).get(self._low_level_obs_key, None)
                if low_obs is None:
                    raise RuntimeError(
                        f"Missing low-level observation '{self._low_level_obs_key}'. "
                        "Add it to policy_obs_dict or enable build_low_level_obs_from_state."
                    )
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
        self._last_low_action[env_ids] = 0.0
        self._base_command[env_ids] = self._base_mid
        self._prev_base_command[env_ids] = self._base_mid
        self._prev_prev_base_command[env_ids] = self._base_mid
        default_pos = self._asset.data.default_joint_pos
        self._arm_target[env_ids] = default_pos[env_ids][:, self._arm_joint_ids]
        if self._num_gripper_actions > 0:
            self._gripper_target[env_ids] = default_pos[env_ids][:, self._gripper_joint_ids]
        if self._gripper_mimic_joint_ids:
            self._gripper_mimic_target[env_ids] = default_pos[env_ids][:, self._gripper_mimic_joint_ids]
        self._leg_target[env_ids] = default_pos[env_ids][:, self._leg_joint_ids]
        reset_obs = self._build_low_level_obs()
        self._low_obs_history[env_ids] = reset_obs[env_ids].unsqueeze(1).repeat(1, self._low_level_history_length, 1)
        self._sim_counter_since_low_update = self._low_level_decimation


@configclass
class HighLevelPickPlaceActionCfg(ActionTermCfg):
    """High-level action split into base command and arm joint command."""

    class_type: type[ActionTerm] = HighLevelPickPlaceAction

    leg_joint_names: list[str] = MISSING
    arm_joint_names: list[str] = MISSING
    gripper_joint_names: list[str] = ()
    gripper_mimic_joint_names: list[str] = ()
    gripper_mimic_multipliers: tuple[float, ...] = ()
    gripper_mimic_offsets: tuple[float, ...] = ()
    gripper_target_range: tuple[float, float] | None = None
    gripper_binary_action: bool = False
    gripper_binary_threshold: float = 0.0
    gripper_closed_target: float = 0.0
    gripper_open_target: float = 0.1
    base_command_low: tuple[float, float, float, float, float] = (-0.6, -0.4, -0.8, 0.25, -0.25)
    base_command_high: tuple[float, float, float, float, float] = (1.0, 0.4, 0.8, 0.45, 0.25)
    arm_action_scale: float | tuple[float, ...] = 0.25
    gripper_action_scale: float = 0.03
    use_default_arm_offset: bool = True
    low_level_policy_path: str | None = None
    low_level_policy_format: str = "auto"
    low_level_actor_cfg: dict | None = None
    low_level_vae_cfg: dict | None = None
    low_level_obs_key: str = "low_actor_obs"
    low_level_obs_dim: int = 47
    low_level_history_length: int = 5
    low_level_action_scale: float = 0.25
    low_level_decimation: int = 4
    build_low_level_obs_from_state: bool = True
    low_level_onnx_use_cuda: bool = True
    low_level_base_ang_vel_scale: float = 0.25
    low_level_joint_vel_scale: float = 0.05
    low_level_last_action_scale: float = 0.25
    low_level_command_scale: tuple[float, float, float, float, float] = (2.0, 2.0, 0.25, 2.0, 1.0)
    zero_leg_action_without_policy: bool = True
    command_smoothing_alpha: float = 1.0
    arm_target_smoothing_alpha: float = 1.0
    arm_target_max_delta: float | None = None
    preserve_order: bool = True
