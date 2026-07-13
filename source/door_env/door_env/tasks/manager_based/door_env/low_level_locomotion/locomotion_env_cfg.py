"""Environment config for hierarchical PiperL pick-and-place."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import rl_sim_env.tasks.manager_based.common.mdp as mdp
from rl_sim_env.tasks.manager_based.common.pick_place import (
    HighLevelPickPlaceActionCfg,
    arm_action_rate_l2,
    arm_joint_vel_l2,
    arm_joint_limit_margin,
    arm_joint_torque_l2,
    arm_nominal_pose_l2,
    base_heading_to_place,
    base_height_l2,
    base_orientation_l2,
    base_height_pitch_command_l2,
    base_planar_velocity_l2,
    base_pitch_roll_height,
    base_retreat,
    base_roll_pitch_ang_vel_l2,
    base_to_place_exp,
    base_to_pick_stance_exp,
    base_vertical_vel_l2,
    body_command_acc_l2,
    body_command_norm_l2,
    body_command_rate_l2,
    complete_success,
    ee_object_relative_velocity_l2,
    ee_object_contact,
    ee_pose_in_base,
    ee_retreat,
    ee_to_object_exp,
    ee_to_object_shaped,
    ee_to_pick_progress,
    ee_to_place_progress,
    ee_to_place_shaped,
    ee_to_target_shaped,
    excessive_height_change_l2,
    excessive_pitch_l2,
    gripper_release,
    gripper_close_near_object,
    gripper_hold_after_place_penalty,
    grasp_stable_progress,
    grasp_hold_failure_penalty,
    grasp_hold_success,
    grasp_success,
    grasping_success_shaped,
    high_action_acc_l2,
    high_action_rate_l2,
    high_level_base_command,
    high_level_previous_action,
    low_level_last_action,
    object_position_in_base,
    object_below_place_penalty,
    object_below_pick_penalty,
    object_fallen,
    object_fallen_penalty,
    object_lift_progress,
    object_hover_over_place_penalty,
    object_on_place_height_success,
    gripper_hold_near_place_penalty,
    object_stability_l2,
    object_tilt_near_place_l2,
    object_to_place_exp,
    object_to_place_progress,
    object_to_place_shaped,
    object_velocity_in_base,
    phase_index,
    phase_progress,
    phase_transition_bonus,
    pick_reached_success,
    pick_position_in_base,
    place_position_in_base,
    place_success,
    post_place_still_success,
    reset_physical_pick_place_scene,
    reset_robot_and_physical_pick_place_scene,
    support_clearance_l2,
    target_position_in_base,
    update_pick_place_phase,
    velocity_command_acc_l2,
    velocity_command_norm_l2,
    velocity_command_rate_l2,
    visualize_pick_place_targets,
    phase2_hold_success,
    virtual_complete_success,
    virtual_place_success_hold,
)
from rl_sim_env.tasks.manager_based.common.sensor.frame_transform_config import create_body_frame_transform_cfg
from rl_sim_env.tasks.manager_based.common.sensor.ray_caster_config import BLIND_HEIGHT_SCANNER_CFG
from rl_sim_env.tasks.manager_based.common.terrain.config import LOCOMOTION_PLANE_CFG
from rl_sim_env.tasks.manager_based.locomotion.locomotion_base_env_cfg import LocomotionEnvCfg
from rl_sim_env.tasks.manager_based.locomotion.locomotion_base_env_cfg import MySceneCfg

from .config_summary import (
    ConfigSummary,
    OBJECT_CFG,
    PICK_SUPPORT_CFG,
    PLACE_SUPPORT_CFG,
    ROBOT_ARM_JOINT_NAMES,
    ROBOT_BASE_LINK,
    ROBOT_CFG,
    ROBOT_EE_BODY_NAME,
    ROBOT_FOOT_NAMES,
    ROBOT_GRIPPER_MIMIC_JOINT_NAMES,
    ROBOT_GRIPPER_JOINT_NAMES,
    ROBOT_JOINT_NAMES,
)


def zero_padding(env, dim: int = 1):
    import torch

    return torch.zeros((env.num_envs, int(dim)), device=env.device)


@configclass
class PickPlaceSceneCfg(MySceneCfg):
    object: RigidObjectCfg = None
    pick_support: RigidObjectCfg = None
    place_support: RigidObjectCfg = None


@configclass
class LocomotionPiperLPickPlaceEnvCfg(LocomotionEnvCfg):
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=4096, env_spacing=10.0)

    def __post_init__(self):
        self.config_summary = ConfigSummary
        num_envs = self.config_summary.env.num_envs

        self.decimation = self.config_summary.general.decimation
        self.episode_length_s = self.config_summary.general.episode_length_s
        self.is_finite_horizon = self.config_summary.general.is_finite_horizon

        self.scene.num_envs = num_envs
        self.scene.robot = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.object = OBJECT_CFG if self.config_summary.task.use_dynamic_object else None
        self.scene.pick_support = PICK_SUPPORT_CFG
        self.scene.place_support = PLACE_SUPPORT_CFG
        self.scene.terrain = LOCOMOTION_PLANE_CFG

        self.scene.height_scanner = BLIND_HEIGHT_SCANNER_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK
        self.scene.height_scanner.update_period = self.decimation * self.config_summary.sim.dt
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ROBOT_BASE_LINK,
            history_length=1,
            track_air_time=False,
        )
        self.scene.contact_forces.update_period = self.config_summary.sim.dt
        self.scene.frame_transform = create_body_frame_transform_cfg(ROBOT_BASE_LINK, ROBOT_FOOT_NAMES)

        self.sim.dt = self.config_summary.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # High-level task command is represented by the action term, not CommandManager.
        self.commands.base_command = None
        self.actions.joint_pos = None
        self.actions.high_level = HighLevelPickPlaceActionCfg(
            asset_name="robot",
            leg_joint_names=ROBOT_JOINT_NAMES,
            arm_joint_names=ROBOT_ARM_JOINT_NAMES,
            gripper_joint_names=ROBOT_GRIPPER_JOINT_NAMES,
            gripper_mimic_joint_names=ROBOT_GRIPPER_MIMIC_JOINT_NAMES,
            gripper_mimic_multipliers=(0.5, -0.5),
            gripper_mimic_offsets=(0.0, 0.0),
            gripper_target_range=(0.0, 0.1),
            gripper_binary_action=self.config_summary.control.gripper_binary_action,
            gripper_binary_threshold=self.config_summary.control.gripper_binary_threshold,
            gripper_closed_target=self.config_summary.control.gripper_closed_target,
            gripper_open_target=self.config_summary.control.gripper_open_target,
            base_command_low=self.config_summary.control.base_command_low,
            base_command_high=self.config_summary.control.base_command_high,
            arm_action_scale=self.config_summary.control.arm_action_scale,
            gripper_action_scale=self.config_summary.control.gripper_action_scale,
            low_level_policy_path=self.config_summary.control.low_level_policy_path,
            low_level_policy_format=self.config_summary.control.low_level_policy_format,
            low_level_actor_cfg=self.config_summary.control.low_level_actor_cfg,
            low_level_vae_cfg=self.config_summary.control.low_level_vae_cfg,
            low_level_obs_dim=self.config_summary.control.low_level_obs_dim,
            low_level_history_length=self.config_summary.control.low_level_history_length,
            low_level_action_scale=self.config_summary.control.low_level_action_scale,
            low_level_decimation=self.config_summary.control.low_level_decimation,
            low_level_onnx_use_cuda=self.config_summary.control.low_level_onnx_use_cuda,
            low_level_base_ang_vel_scale=self.config_summary.control.low_level_base_ang_vel_scale,
            low_level_joint_vel_scale=self.config_summary.control.low_level_joint_vel_scale,
            low_level_last_action_scale=self.config_summary.control.low_level_last_action_scale,
            low_level_command_scale=self.config_summary.control.low_level_command_scale,
            command_smoothing_alpha=self.config_summary.control.command_smoothing_alpha,
            arm_target_smoothing_alpha=self.config_summary.control.arm_target_smoothing_alpha,
            arm_target_max_delta=self.config_summary.control.arm_target_max_delta,
        )

        self._configure_observations()
        self._configure_events()
        self._configure_terminations()
        self._configure_rewards()

        if self.config_summary.task.base_contact_termination_enable:
            self.terminations.base_contact.params["sensor_cfg"].body_names = ROBOT_BASE_LINK
            self.terminations.base_contact.params["threshold"] = self.config_summary.task.base_contact_force_threshold
        else:
            self.terminations.base_contact = None
        self.terminations.bad_orientation.params["asset_cfg"].body_names = ROBOT_BASE_LINK
        self.curriculum.terrain_levels_vel = None
        self.curriculum.lin_vel_x_command_threshold = None
        self.curriculum.ang_vel_z_command_threshold = None
        self.curriculum.ee_external_force_threshold = None
        self.curriculum.pick_place_target_range = CurrTerm(func=mdp.pick_place_target_range)

    def _configure_observations(self):
        obs = self.observations.ground_truth_obs
        obs.arm_joint_pos_rel_gt = ObsTerm(func=mdp.joint_pos_rel)
        obs.arm_joint_vel_rel_gt = ObsTerm(func=mdp.joint_vel_rel)
        obs.base_pitch_roll_height_gt = ObsTerm(func=base_pitch_roll_height)
        obs.ee_pose_b_gt = ObsTerm(func=ee_pose_in_base)
        if self.config_summary.task.use_dynamic_object:
            obs.object_pos_b_gt = ObsTerm(func=object_position_in_base)
            obs.object_vel_b_gt = ObsTerm(func=object_velocity_in_base)
        else:
            obs.object_pos_b_gt = ObsTerm(func=target_position_in_base)
            obs.object_vel_b_gt = ObsTerm(func=zero_padding, params={"dim": 3})
        obs.pick_pos_b_gt = ObsTerm(func=pick_position_in_base)
        obs.place_pos_b_gt = ObsTerm(func=place_position_in_base)
        obs.phase_gt = ObsTerm(func=phase_index)
        obs.high_action_gt = ObsTerm(func=high_level_previous_action)
        obs.high_base_command_gt = ObsTerm(func=high_level_base_command)
        obs.low_action_gt = ObsTerm(func=low_level_last_action, params={"dim": len(ROBOT_JOINT_NAMES)})
        obs.gt_padding_1 = ObsTerm(func=zero_padding, params={"dim": 1})
        obs.gt_padding_2 = ObsTerm(func=zero_padding, params={"dim": 1})
        nad = self.observations.noise_and_delay_obs
        nad.base_ang_vel_noisy = ObsTerm(func=mdp.base_ang_vel)
        nad.projected_gravity_noisy = ObsTerm(func=mdp.projected_gravity)
        nad.joint_pos_rel_noisy = ObsTerm(func=mdp.joint_pos_rel)
        nad.joint_vel_rel_noisy = ObsTerm(func=mdp.joint_vel_rel)
        nad.arm_joint_pos_rel_noisy = ObsTerm(func=mdp.joint_pos_rel)
        nad.arm_joint_vel_rel_noisy = ObsTerm(func=mdp.joint_vel_rel)
        nad.base_pitch_roll_height_noisy = ObsTerm(func=base_pitch_roll_height)
        nad.ee_pose_b_noisy = ObsTerm(func=ee_pose_in_base)
        if self.config_summary.task.use_dynamic_object:
            nad.object_pos_b_noisy = ObsTerm(func=object_position_in_base)
            nad.object_vel_b_noisy = ObsTerm(func=object_velocity_in_base)
        else:
            nad.object_pos_b_noisy = ObsTerm(func=target_position_in_base)
            nad.object_vel_b_noisy = ObsTerm(func=zero_padding, params={"dim": 3})
        nad.pick_pos_b_noisy = ObsTerm(func=pick_position_in_base)
        nad.place_pos_b_noisy = ObsTerm(func=place_position_in_base)
        nad.nad_padding_1 = ObsTerm(func=zero_padding, params={"dim": 1})
        self.observations.noise_and_delay_obs.enable_corruption = False
        self.observations.amp_obs.amp_padding_1 = ObsTerm(func=zero_padding, params={"dim": 1})

        to_drop = {
            "concatenate_terms",
            "concatenate_dim",
            "enable_corruption",
            "history_length",
            "flatten_history_dim",
        }
        invalid_obs_group_keys = list(self.observations.__dict__.keys() - self.config_summary.observation.obs_term_dict.keys())
        for key in invalid_obs_group_keys:
            self.observations.__dict__[key] = None

        for group_key, group_value in self.config_summary.observation.obs_term_dict.items():
            invalid_obs_term_keys = list(self.observations.__dict__[group_key].__dict__.keys() - group_value.keys())
            invalid_obs_term_keys[:] = [x for x in invalid_obs_term_keys if x not in to_drop]
            for key in invalid_obs_term_keys:
                self.observations.__dict__[group_key].__dict__[key] = None
            for key, value in group_value.items():
                term = self.observations.__dict__[group_key].__dict__[key]
                if "scale" in value:
                    term.scale = value["scale"]
                if "noise" in value:
                    term.noise = value["noise"]
                if "clip" in value:
                    term.clip = value["clip"]
                if "params" in value:
                    for k, v in value["params"].items():
                        term.params[k] = v
        self.observations.ground_truth_obs.enable_corruption = False
        self.observations.noise_and_delay_obs.enable_corruption = bool(
            getattr(self.config_summary.observation, "enable_actor_obs_noise", False)
        )

    def _configure_events(self):
        self.events.reset_base = EventTerm(func=reset_robot_and_physical_pick_place_scene, mode="reset")
        self.events.update_pick_place_phase = EventTerm(
            func=update_pick_place_phase,
            mode="interval",
            interval_range_s=(0.02, 0.02),
        )
        self.events.visualize_pick_place_targets = EventTerm(
            func=visualize_pick_place_targets,
            mode="interval",
            interval_range_s=(0.1, 0.1),
        )

        event_config = dict(self.config_summary.event.config_dict)
        if not event_config.get("reset_base", {}).get("params", {}).get("debug_vis", False):
            event_config.pop("visualize_pick_place_targets", None)

        invalid_events_keys = list(self.events.__dict__.keys() - event_config.keys())
        for key in invalid_events_keys:
            self.events.__dict__[key] = None
        for key, value in event_config.items():
            self.events.__dict__[key].mode = value["mode"]
            if "interval_range_s" in value:
                self.events.__dict__[key].interval_range_s = value["interval_range_s"]
            if "params" in value:
                for k, v in value["params"].items():
                    self.events.__dict__[key].params[k] = v

    def _configure_terminations(self):
        if self.config_summary.task.use_dynamic_object:
            self.terminations.virtual_place_success = None
            self.terminations.object_fallen = DoneTerm(
                func=object_fallen,
                params={
                    "object_cfg": SceneEntityCfg("object"),
                    "min_height": self.config_summary.task.object_fallen_min_height,
                    "phases": self.config_summary.task.object_fallen_termination_phases,
                },
            )
        else:
            self.terminations.object_fallen = None
            self.terminations.virtual_place_success = DoneTerm(func=virtual_place_success_hold)
            self.terminations.virtual_place_success.params.update(
                {
                    "ee_asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME),
                    "hold_time_s": self.config_summary.task.phase2_hold_time_s,
                    "ee_place_threshold": self.config_summary.task.phase2_ee_place_threshold,
                    "base_place_threshold": self.config_summary.task.phase2_base_place_threshold,
                    "heading_threshold": self.config_summary.task.phase2_heading_threshold,
                    "roll_limit": self.config_summary.task.phase2_roll_limit,
                    "pitch_limit": self.config_summary.task.phase2_pitch_limit,
                }
            )

    def _configure_rewards(self):
        self.rewards.base_to_pick_stance = RewTerm(func=base_to_pick_stance_exp, weight=0.0)
        self.rewards.ee_to_object = RewTerm(func=ee_to_object_exp, weight=0.0)
        self.rewards.ee_to_object_shaped = RewTerm(func=ee_to_object_shaped, weight=0.0)
        self.rewards.ee_to_target_shaped = RewTerm(func=ee_to_target_shaped, weight=0.0)
        self.rewards.ee_to_pick_progress = RewTerm(func=ee_to_pick_progress, weight=0.0)
        self.rewards.pick_reached_success = RewTerm(func=pick_reached_success, weight=0.0)
        self.rewards.ee_to_place_shaped = RewTerm(func=ee_to_place_shaped, weight=0.0)
        self.rewards.ee_to_place_progress = RewTerm(func=ee_to_place_progress, weight=0.0)
        self.rewards.ee_object_contact = RewTerm(func=ee_object_contact, weight=0.0)
        self.rewards.grasp_success = RewTerm(func=grasp_success, weight=0.0)
        self.rewards.grasp_hold_success = RewTerm(func=grasp_hold_success, weight=0.0)
        self.rewards.grasp_hold_failure_penalty = RewTerm(func=grasp_hold_failure_penalty, weight=0.0)
        self.rewards.grasping_success_shaped = RewTerm(func=grasping_success_shaped, weight=0.0)
        self.rewards.grasp_stable_progress = RewTerm(func=grasp_stable_progress, weight=0.0)
        self.rewards.object_lift_progress = RewTerm(func=object_lift_progress, weight=0.0)
        self.rewards.gripper_close_near_object = RewTerm(func=gripper_close_near_object, weight=0.0)
        self.rewards.object_below_pick_penalty = RewTerm(func=object_below_pick_penalty, weight=0.0)
        self.rewards.object_fallen_penalty = RewTerm(func=object_fallen_penalty, weight=0.0)
        self.rewards.base_heading_to_place = RewTerm(func=base_heading_to_place, weight=0.0)
        self.rewards.object_to_place = RewTerm(func=object_to_place_exp, weight=0.0)
        self.rewards.object_to_place_shaped = RewTerm(func=object_to_place_shaped, weight=0.0)
        self.rewards.object_to_place_progress = RewTerm(func=object_to_place_progress, weight=0.0)
        self.rewards.base_to_place = RewTerm(func=base_to_place_exp, weight=0.0)
        self.rewards.place_success = RewTerm(func=place_success, weight=0.0)
        self.rewards.gripper_release = RewTerm(func=gripper_release, weight=0.0)
        self.rewards.object_on_place_height_success = RewTerm(func=object_on_place_height_success, weight=0.0)
        self.rewards.object_below_place_penalty = RewTerm(func=object_below_place_penalty, weight=0.0)
        self.rewards.gripper_hold_after_place_penalty = RewTerm(func=gripper_hold_after_place_penalty, weight=0.0)
        self.rewards.gripper_hold_near_place_penalty = RewTerm(func=gripper_hold_near_place_penalty, weight=0.0)
        self.rewards.object_hover_over_place_penalty = RewTerm(func=object_hover_over_place_penalty, weight=0.0)
        self.rewards.post_place_still_success = RewTerm(func=post_place_still_success, weight=0.0)
        self.rewards.base_retreat = RewTerm(func=base_retreat, weight=0.0)
        self.rewards.ee_retreat = RewTerm(func=ee_retreat, weight=0.0)
        self.rewards.complete_success = RewTerm(func=complete_success, weight=0.0)
        self.rewards.phase2_hold_success = RewTerm(func=phase2_hold_success, weight=0.0)
        self.rewards.phase_progress = RewTerm(func=phase_progress, weight=0.0)
        self.rewards.phase_transition_bonus = RewTerm(func=phase_transition_bonus, weight=0.0)
        self.rewards.virtual_complete_success = RewTerm(func=virtual_complete_success, weight=0.0)
        self.rewards.high_action_rate = RewTerm(func=high_action_rate_l2, weight=0.0)
        self.rewards.high_action_acc = RewTerm(func=high_action_acc_l2, weight=0.0)
        self.rewards.base_contact_force = RewTerm(func=mdp.contact_forces_l2, weight=0.0)
        self.rewards.base_orientation = RewTerm(func=base_orientation_l2, weight=0.0)
        self.rewards.base_height_final = RewTerm(func=base_height_l2, weight=0.0)
        self.rewards.base_orientation_final = RewTerm(func=base_orientation_l2, weight=0.0)
        self.rewards.arm_joint_vel = RewTerm(func=arm_joint_vel_l2, weight=0.0)
        self.rewards.post_place_arm_joint_vel = RewTerm(func=arm_joint_vel_l2, weight=0.0)
        self.rewards.arm_nominal_pose_final = RewTerm(func=arm_nominal_pose_l2, weight=0.0)
        self.rewards.arm_action_rate = RewTerm(func=arm_action_rate_l2, weight=0.0)
        self.rewards.arm_torque = RewTerm(func=arm_joint_torque_l2, weight=0.0)
        self.rewards.arm_nominal_pose = RewTerm(func=arm_nominal_pose_l2, weight=0.0)
        self.rewards.arm_joint_limit = RewTerm(func=arm_joint_limit_margin, weight=0.0)
        self.rewards.base_vertical_vel = RewTerm(func=base_vertical_vel_l2, weight=0.0)
        self.rewards.base_roll_pitch_ang_vel = RewTerm(func=base_roll_pitch_ang_vel_l2, weight=0.0)
        self.rewards.base_planar_velocity = RewTerm(func=base_planar_velocity_l2, weight=0.0)
        self.rewards.velocity_command_norm = RewTerm(func=velocity_command_norm_l2, weight=0.0)
        self.rewards.base_height_pitch_command = RewTerm(func=base_height_pitch_command_l2, weight=0.0)
        self.rewards.velocity_command_rate = RewTerm(func=velocity_command_rate_l2, weight=0.0)
        self.rewards.velocity_command_acc = RewTerm(func=velocity_command_acc_l2, weight=0.0)
        self.rewards.body_command_norm = RewTerm(func=body_command_norm_l2, weight=0.0)
        self.rewards.body_command_rate = RewTerm(func=body_command_rate_l2, weight=0.0)
        self.rewards.body_command_acc = RewTerm(func=body_command_acc_l2, weight=0.0)
        self.rewards.excessive_pitch = RewTerm(func=excessive_pitch_l2, weight=0.0)
        self.rewards.excessive_height_change = RewTerm(func=excessive_height_change_l2, weight=0.0)
        self.rewards.support_clearance = RewTerm(func=support_clearance_l2, weight=0.0)
        self.rewards.object_stability = RewTerm(func=object_stability_l2, weight=0.0)
        self.rewards.object_tilt_near_place = RewTerm(func=object_tilt_near_place_l2, weight=0.0)
        self.rewards.ee_object_relative_velocity = RewTerm(func=ee_object_relative_velocity_l2, weight=0.0)

        reward_config = dict(self.config_summary.reward.config_dict)
        if not self.config_summary.task.use_dynamic_object:
            dynamic_reward_keys = {
                "ee_to_object",
                "ee_to_object_shaped",
                "ee_object_contact",
                "grasp_success",
                "grasp_hold_success",
                "grasp_hold_failure_penalty",
                "grasping_success_shaped",
                "grasp_stable_progress",
                "object_lift_progress",
                "gripper_close_near_object",
                "object_fallen_penalty",
                "object_to_place",
                "object_to_place_shaped",
                "object_to_place_progress",
                "place_success",
                "gripper_release",
                "object_on_place_height_success",
                "object_below_place_penalty",
                "gripper_hold_after_place_penalty",
                "gripper_hold_near_place_penalty",
                "object_hover_over_place_penalty",
                "post_place_still_success",
                "base_retreat",
                "ee_retreat",
                "complete_success",
                "object_stability",
                "object_tilt_near_place",
                "ee_object_relative_velocity",
            }
            for key in dynamic_reward_keys:
                reward_config.pop(key, None)

        invalid_rewards_keys = list(self.rewards.__dict__.keys() - reward_config.keys())
        for key in invalid_rewards_keys:
            self.rewards.__dict__[key] = None
        for key, value in reward_config.items():
            self.rewards.__dict__[key].weight = value["weight"]
            if "params" in value:
                for k, v in value["params"].items():
                    self.rewards.__dict__[key].params[k] = v


@configclass
class LocomotionPiperLPickPlaceEnvCfg_PLAY(LocomotionPiperLPickPlaceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.env_spacing = 10.0
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            side = int(math.ceil(math.sqrt(self.scene.num_envs)))
            self.scene.terrain.terrain_generator.num_rows = side
            self.scene.terrain.terrain_generator.num_cols = side
            self.scene.terrain.terrain_generator.curriculum = False
        self.observations.ground_truth_obs.enable_corruption = False
        self.curriculum.pick_place_target_range = None
        self.events.reset_base.params.update(
            {
                "pick_x_range": self.config_summary.task.pick_x_range_final,
                "pick_y_range": self.config_summary.task.pick_y_range_final,
                "pick_z_range": self.config_summary.task.pick_z_range_final,
                "place_x_range": self.config_summary.task.place_x_range_final,
                "place_y_range": self.config_summary.task.place_y_range_final,
                "place_z_range": self.config_summary.task.place_z_range_final,
            }
        )


class LocomotionPiperLPickPlaceEnvCfg_DEBUG_PLAY(LocomotionPiperLPickPlaceEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.terminations.bad_orientation = None
