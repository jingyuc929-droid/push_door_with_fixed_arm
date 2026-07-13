"""Teacher-side interfaces reserved for future DoorBot Student-RNN distillation."""

from __future__ import annotations

import os
import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP, EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups
from torch.distributions import Normal


POLICY_OBS_SCHEMA = (
    ("robot_arm_joint_pos", 6), ("robot_arm_joint_vel", 6),
    ("last_applied_arm_delta", 6), ("arm_q_des_error", 6),
    ("last_high_base_action", 3), ("last_arm_action", 6), ("high_base_command", 3),
    ("base_velocity_b", 5), ("projected_gravity_b", 3), ("base_height", 1),
    ("base_to_doorway_center_b_xy", 2), ("doorway_forward_axis_b_xy", 2),
    ("ee_to_handle_target_b", 3), ("handle_target_position_b", 3),
    ("door_panel_forward_axis_b_xy", 2),
)

PRIVILEGED_STATE_SCHEMA = (
    ("ee_pos_in_handle", 3), ("ee_quat_err_in_handle", 4),
    ("handle_joint_pos", 1), ("handle_joint_vel", 1),
    ("door_joint_pos", 1), ("door_joint_vel", 1),
    ("unlock_state", 1), ("stage_id", 1),
    ("body_door_contact_force_norm", 1), ("leg_door_contact_force_norm", 1),
    ("body_door_frame_contact_force_norm", 1), ("leg_door_frame_contact_force_norm", 1),
)

DECODER_CONTINUOUS_SCHEMA = (
    ("handle_joint_pos", 1), ("handle_joint_vel", 1),
    ("door_joint_pos", 1), ("door_joint_vel", 1),
)
DECODER_DISCRETE_SCHEMA = (("unlock_state", 1), ("stage_id", 1), ("transition_flags", 5))
TRANSITION_FLAG_SCHEMA = (
    "newly_grasped", "newly_unlocked", "entered_initial_push", "entered_push_follow", "entered_traverse"
)


class DoorBotTeacherExportModule(nn.Module):
    """Exportable two-input graph for the privileged-latent Teacher actor."""

    def __init__(self, policy):
        super().__init__()
        self.observation_normalizer = copy.deepcopy(policy.actor_obs_normalizer)
        self.privileged_normalizer = copy.deepcopy(policy.privileged_normalizer)
        self.privileged_encoder = copy.deepcopy(policy.privileged_encoder)
        self.actor = copy.deepcopy(policy.actor)

    def forward(self, policy_obs_clean: torch.Tensor, privileged_state: torch.Tensor):
        clean = self.observation_normalizer(policy_obs_clean)
        privileged = self.privileged_normalizer(privileged_state)
        z_priv = self.privileged_encoder(privileged)
        return self.actor(torch.cat((clean, z_priv), dim=-1))


def export_doorbot_teacher(policy, path: str, jit_filename="policy.pt", onnx_filename="policy.onnx"):
    """Export the Teacher with explicit clean-observation and privileged-state inputs."""
    os.makedirs(path, exist_ok=True)
    module = DoorBotTeacherExportModule(policy).cpu().eval()
    clean_dim = sum(dim for _, dim in POLICY_OBS_SCHEMA)
    privileged_dim = sum(dim for _, dim in PRIVILEGED_STATE_SCHEMA)
    clean = torch.zeros(1, clean_dim, dtype=torch.float32)
    privileged = torch.zeros(1, privileged_dim, dtype=torch.float32)

    traced = torch.jit.trace(module, (clean, privileged))
    traced.save(os.path.join(path, jit_filename))
    torch.onnx.export(
        module,
        (clean, privileged),
        os.path.join(path, onnx_filename),
        input_names=["policy_obs_clean", "privileged_state"],
        output_names=["teacher_action_raw"],
        dynamic_axes={
            "policy_obs_clean": {0: "batch"},
            "privileged_state": {0: "batch"},
            "teacher_action_raw": {0: "batch"},
        },
        opset_version=17,
    )


class DoorBotPrivilegedActorCritic(ActorCritic):
    """Teacher actor using clean deployable observations plus an encoded privileged latent."""

    def __init__(
        self, obs, obs_groups, num_actions, z_priv_dim=16, privileged_encoder_hidden_dims=(64, 32),
        clean_obs_group="policy_obs_clean", privileged_obs_group="privileged_state",
        actor_obs_normalization=True, critic_obs_normalization=True,
        privileged_obs_normalization=True, actor_hidden_dims=(256, 256, 128),
        critic_hidden_dims=(256, 256, 128), activation="elu", init_noise_std=1.0,
        noise_std_type="scalar", **kwargs,
    ):
        nn.Module.__init__(self)
        self.obs_groups = obs_groups
        self.clean_obs_group = clean_obs_group
        self.privileged_obs_group = privileged_obs_group
        self.z_priv_dim = int(z_priv_dim)
        clean_dim = obs[clean_obs_group].shape[-1]
        priv_dim = obs[privileged_obs_group].shape[-1]
        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.privileged_obs_normalization = bool(privileged_obs_normalization)
        self.actor_obs_normalizer = EmpiricalNormalization(clean_dim) if actor_obs_normalization else nn.Identity()
        self.privileged_normalizer = EmpiricalNormalization(priv_dim) if privileged_obs_normalization else nn.Identity()
        self.critic_obs_normalizer = EmpiricalNormalization(clean_dim + priv_dim) if critic_obs_normalization else nn.Identity()
        self.privileged_encoder = MLP(priv_dim, self.z_priv_dim, list(privileged_encoder_hidden_dims), activation)
        self.actor = MLP(clean_dim + self.z_priv_dim, num_actions, list(actor_hidden_dims), activation)
        self.critic = MLP(clean_dim + priv_dim, 1, list(critic_hidden_dims), activation)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self.distribution = None
        self.last_z_priv = None
        Normal.set_default_validate_args(False)

    def _teacher_actor_input(self, obs):
        clean = self.actor_obs_normalizer(obs[self.clean_obs_group])
        privileged = self.privileged_normalizer(obs[self.privileged_obs_group])
        self.last_z_priv = self.privileged_encoder(privileged)
        return torch.cat((clean, self.last_z_priv), dim=-1)

    def act(self, obs, **kwargs):
        actor_input = self._teacher_actor_input(obs)
        self.update_distribution(actor_input)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(self._teacher_actor_input(obs))

    def evaluate(self, obs, **kwargs):
        critic_input = torch.cat((obs[self.clean_obs_group], obs[self.privileged_obs_group]), dim=-1)
        return self.critic(self.critic_obs_normalizer(critic_input))

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(obs[self.clean_obs_group])
        if self.privileged_obs_normalization:
            self.privileged_normalizer.update(obs[self.privileged_obs_group])
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(
                torch.cat((obs[self.clean_obs_group], obs[self.privileged_obs_group]), dim=-1)
            )


class DoorBotPPO(PPO):
    """PPO with a non-training, time-major auxiliary rollout interface."""

    def __init__(self, *args, collect_distillation_rollout=False, distillation_rollout_steps=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_distillation_rollout = bool(collect_distillation_rollout)
        self.distillation_rollout_steps = int(distillation_rollout_steps)
        self._distillation_buffers = None
        self._distillation_step = 0
        self.last_distillation_rollout = None

    def process_env_step(self, obs, rewards, dones, extras):
        auxiliary = extras.get("distillation") if self.collect_distillation_rollout else None
        if auxiliary is not None:
            if self.policy.last_z_priv is not None:
                auxiliary["z_priv"] = self.policy.last_z_priv.detach()
            if self._distillation_buffers is None:
                self._distillation_buffers = {
                    key: torch.empty(
                        (self.distillation_rollout_steps, *value.shape), device=value.device, dtype=value.dtype
                    )
                    for key, value in auxiliary.items()
                }
            if self._distillation_step >= self.distillation_rollout_steps:
                raise RuntimeError("Distillation rollout exceeded the configured preallocated length.")
            for key, value in auxiliary.items():
                self._distillation_buffers[key][self._distillation_step].copy_(value.detach())
            self._distillation_step += 1
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        if self._distillation_buffers is not None and self._distillation_step:
            # Transfer ownership instead of stacking per-step tensors. A new
            # buffer is allocated only when the next collected rollout starts.
            self.last_distillation_rollout = {
                key: value[: self._distillation_step] for key, value in self._distillation_buffers.items()
            }
            self._distillation_buffers = None
            self._distillation_step = 0
        return super().update()

    def pop_distillation_rollout(self):
        """Return the latest unshuffled [time, env, ...] rollout and release its reference.

        Consumers reconstruct episodes across rollout boundaries with
        ``episode_done_mask``; stage transitions never split or clear a stream.
        """
        rollout = self.last_distillation_rollout
        self.last_distillation_rollout = None
        return rollout


class DoorBotDistillationVecEnvWrapper(RslRlVecEnvWrapper):
    """Adds reset-aware history and distillation tensors without changing environment dynamics."""

    def __init__(
        self, env, clip_actions=None, history_length=32, transition_thresholds=None,
        collect_distillation_rollout=False,
    ):
        super().__init__(env, clip_actions)
        self.history_length = int(history_length)
        self.collect_distillation_rollout = bool(collect_distillation_rollout)
        self.transition_thresholds = transition_thresholds or {
            "initial_push": 0.02, "push_follow": 0.10, "traverse": 0.70
        }
        obs = self.get_observations()
        self._cached_obs = obs
        noisy = obs["policy_obs_noisy"]
        self.history = torch.zeros(
            self.num_envs, self.history_length, noisy.shape[-1], device=self.device, dtype=noisy.dtype
        )
        self.history_valid_mask = torch.zeros(
            self.num_envs, self.history_length, device=self.device, dtype=torch.bool
        )
        self.history[:, 0] = noisy
        self.history_valid_mask[:, 0] = True
        self._history_write_index = 1 % self.history_length
        self.distillation_constants = {
            "command_limits": self._command_limits(),
            "history_source": "policy_obs_noisy",
            "history_layout": "ring_buffer",
        }
        if self.collect_distillation_rollout:
            self._prev_stage = self._stage_id()
            self._prev_door_angle = self._door_angle()
            self._prev_forward_progress = self._forward_progress()

    @staticmethod
    def _inject_noisy_observation(obs):
        """Use clean observation storage directly until a noise model is configured."""
        if "policy_obs_noisy" not in obs.keys():
            obs["policy_obs_noisy"] = obs["policy_obs_clean"]
        return obs

    def get_observations(self):
        return self._inject_noisy_observation(super().get_observations())

    @property
    def history_write_index(self):
        """Index where the next sample will be written in the physical ring buffer."""
        return self._history_write_index

    def ordered_history(self):
        """Materialize oldest-to-newest history only when a future Student requests it."""
        index = self._history_write_index
        return torch.cat((self.history[:, index:], self.history[:, :index]), dim=1)

    def _base_env(self):
        return self.unwrapped

    def _action_term(self):
        return self._base_env().action_manager._terms["high_level_action"]

    def _stage_id(self):
        env = self._base_env()
        grasped = getattr(env, "_grasp_success_given", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        unlocked = getattr(env, "_door_lock_mode", torch.zeros(self.num_envs, dtype=torch.long, device=self.device)) == 2
        stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        stage = torch.where(grasped & (~unlocked), torch.ones_like(stage), stage)
        return torch.where(unlocked, torch.full_like(stage, 2), stage)

    def _door_joints(self):
        door = self._base_env().scene["door"]
        handle_id = door.find_joints("handle_joint")[0][0]
        door_id = door.find_joints("door_joint")[0][0]
        return (
            door.data.joint_pos[:, handle_id], door.data.joint_vel[:, handle_id],
            door.data.joint_pos[:, door_id], door.data.joint_vel[:, door_id],
        )

    def _door_angle(self):
        return torch.clamp(self._door_joints()[2], min=0.0)

    def _forward_progress(self):
        env = self._base_env()
        robot = env.scene["robot"]
        door = env.scene["door"]
        center_d = torch.tensor((0.6, 0.0, 0.0), device=self.device).view(1, 3).expand(self.num_envs, 3)
        forward_d = torch.tensor((0.0, -1.0, 0.0), device=self.device).view(1, 3).expand(self.num_envs, 3)
        from .mdp.observations import quat_rotate
        center_w = door.data.root_pos_w + quat_rotate(door.data.root_quat_w, center_d)
        forward_w = quat_rotate(door.data.root_quat_w, forward_d)[:, :2]
        forward_w = forward_w / torch.clamp(torch.linalg.norm(forward_w, dim=-1, keepdim=True), min=1.0e-6)
        return torch.sum((robot.data.root_pos_w[:, :2] - center_w[:, :2]) * forward_w, dim=-1)

    def _applied_action(self):
        term = self._action_term()
        return torch.cat((term.base_command, term.applied_delta), dim=-1)

    def _command_limits(self):
        term = self._action_term()
        max_delta = float(term.cfg.arm_target_max_delta or 0.0)
        low = torch.cat((term._base_low.squeeze(0), -torch.full((6,), max_delta, device=self.device)))
        high = torch.cat((term._base_high.squeeze(0), torch.full((6,), max_delta, device=self.device)))
        low = low.clone(); high = high.clone()
        low[3] = high[3] = float(term.cfg.default_body_height)
        low[4] = high[4] = 0.0
        return torch.stack((low, high), dim=-1)

    def step(self, actions):
        pre_obs = self._cached_obs
        if self.collect_distillation_rollout:
            raw_action = actions.detach()
            pre_history_valid = self.history_valid_mask.clone()
        obs, rewards, dones, extras = super().step(actions)
        obs = self._inject_noisy_observation(obs)
        if self.collect_distillation_rollout:
            current_stage = self._stage_id()
            door_angle = self._door_angle()
            thresholds = self.transition_thresholds
            flags = torch.stack((
                (self._prev_stage == 0) & (current_stage >= 1),
                (self._prev_stage < 2) & (current_stage == 2),
                (self._prev_door_angle < thresholds["initial_push"]) & (door_angle >= thresholds["initial_push"]),
                (self._prev_door_angle < thresholds["push_follow"]) & (door_angle >= thresholds["push_follow"]),
                (self._prev_door_angle < thresholds["traverse"]) & (door_angle >= thresholds["traverse"]),
            ), dim=-1)
            hpos, _, dpos, _ = self._door_joints()
            privileged_pre = pre_obs["privileged_state"]
            extras["distillation"] = {
                "policy_obs_clean": pre_obs["policy_obs_clean"],
                # history_t is an alias defined by distillation_constants, so it is not copied twice.
                "policy_obs_noisy": pre_obs["policy_obs_noisy"],
                "privileged_state": privileged_pre,
                "history_valid_mask": pre_history_valid,
                "episode_done_mask": dones.to(dtype=torch.bool),
                "stage_id": self._prev_stage,
                "stage_one_hot": torch.nn.functional.one_hot(self._prev_stage, num_classes=3).float(),
                "transition_flags": flags,
                "decoder_continuous_targets": privileged_pre[:, 7:11],
                "decoder_discrete_targets": torch.cat((privileged_pre[:, 11:13], flags.float()), dim=-1),
                "handle_joint_pos": hpos,
                "door_joint_pos": dpos,
                "base_forward_progress": self._prev_forward_progress,
                "teacher_action_raw": raw_action,
                "teacher_action_applied": self._applied_action(),
                "action_mask": torch.ones_like(raw_action, dtype=torch.bool),
            }
        noisy = obs["policy_obs_noisy"]
        done = dones.to(dtype=torch.bool)
        self.history[done] = 0.0
        self.history_valid_mask[done] = False
        index = self._history_write_index
        self.history[:, index] = noisy
        self.history_valid_mask[:, index] = True
        self._history_write_index = (index + 1) % self.history_length
        self._cached_obs = obs
        if self.collect_distillation_rollout:
            self._prev_stage = current_stage
            self._prev_door_angle = door_angle
            self._prev_forward_progress = self._forward_progress()
        return obs, rewards, dones, extras


class DoorBotTeacherRunner(OnPolicyRunner):
    """OnPolicyRunner with privileged-latent policy and enriched checkpoint metadata."""

    def _construct_algorithm(self, obs):
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        policy = DoorBotPrivilegedActorCritic(
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
        ).to(self.device)
        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        alg = DoorBotPPO(
            policy, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg,
            collect_distillation_rollout=self.cfg.get("collect_distillation_rollout", False),
            distillation_rollout_steps=self.num_steps_per_env,
        )
        alg.distillation_constants = getattr(self.env, "distillation_constants", {})
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def save(self, path, infos=None):
        super().save(path, infos)
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        policy = self.alg.policy
        cfg = self.cfg
        checkpoint.update({
            "privileged_encoder_state_dict": policy.privileged_encoder.state_dict(),
            "observation_normalizer": policy.actor_obs_normalizer.state_dict(),
            "privileged_normalizer": policy.privileged_normalizer.state_dict(),
            "policy_obs_schema_and_order": POLICY_OBS_SCHEMA,
            "privileged_state_schema_and_order": PRIVILEGED_STATE_SCHEMA,
            "z_priv_dim": policy.z_priv_dim,
            "action_scale_and_clip": self._action_metadata(),
            "fixed_body_height": 0.43,
            "fixed_body_pitch": 0.0,
            "history_length": cfg.get("history_length", 32),
            "stage_transition_thresholds": cfg.get("stage_transition_thresholds", {}),
            "control_dt": float(self.env.unwrapped.step_dt),
        })
        torch.save(checkpoint, path)

    def _action_metadata(self):
        term = self.env.unwrapped.action_manager._terms["high_level_action"]
        return {
            "runner_clip": self.cfg.get("clip_actions"),
            "action_dim": self.env.num_actions,
            "base_command_low": tuple(float(x) for x in term.cfg.base_command_low),
            "base_command_high": tuple(float(x) for x in term.cfg.base_command_high),
            "arm_action_scale": tuple(float(x) for x in term.cfg.arm_action_scale),
            "arm_target_max_delta": float(term.cfg.arm_target_max_delta),
            "stage2_arm_scale": float(term.cfg.stage2_arm_scale),
            "stage2_wrist_scale": float(term.cfg.stage2_wrist_scale),
            "command_smoothing_alpha": float(term.cfg.command_smoothing_alpha),
        }
