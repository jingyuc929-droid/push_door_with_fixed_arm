"""High-level pick-and-place task config for GRQ20 V2D4 + PiperL gripper."""

from __future__ import annotations

import copy
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from rl_algorithms.rsl_rl_wrapper import LocomotionOnPolicyRunnerCfg
from rl_sim_env import RL_SIM_ENV_ROOT_DIR

from rl_sim_env.tasks.manager_based.locomotion.config.grq20_v2d4_piperL_loco_manip.config_summary import (
    AMPDataCfg,
    ROBOT_ARM_JOINT_NAMES,
    ROBOT_BASE_LINK,
    ROBOT_CFG as _BASE_ROBOT_CFG,
    ROBOT_FOOT_NAMES,
    ROBOT_JOINT_NAMES,
    ROBOT_THIGH_NAMES,
    RL_SIM_ENV_CONFIG_SUMMARY_DIR,
)

ROBOT_EE_BODY_NAME = ["piperL_gripper_link1", "piperL_gripper_link2"]
ROBOT_GRIPPER_JOINT_NAMES = ["piperL_gripper"]
ROBOT_GRIPPER_MIMIC_JOINT_NAMES = ["piperL_gripper_joint1", "piperL_gripper_joint2"]
ROBOT_GRIPPER_ACTUATOR_JOINT_NAMES = [
    "piperL_gripper",
    "piperL_gripper_joint1",
    "piperL_gripper_joint2",
]
GRIPPER_ASSET_DIR = os.path.join(RL_SIM_ENV_ROOT_DIR, "data/assets/robots/grq20_v2d4_piperL_front_mount_gripper")

ROBOT_CFG = _BASE_ROBOT_CFG.replace(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{GRIPPER_ASSET_DIR}/grq20_v2d4_piperL_front_mount_gripper.urdf",
        usd_dir=GRIPPER_ASSET_DIR,
        usd_file_name="grq20_v2d4_piperL_front_mount_gripper.usd",
        force_usd_conversion=False,
        make_instanceable=True,
        fix_base=False,
        root_link_name=None,
        link_density=0.0,
        merge_fixed_joints=False,
        convert_mimic_joints_to_normal_joints=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0),
        ),
        collider_type="convex_hull",
        self_collision=False,
        replace_cylinders_with_capsules=False,
        collision_from_visuals=False,
        activate_contact_sensors=True,
        rigid_props=_BASE_ROBOT_CFG.spawn.rigid_props,
        articulation_props=_BASE_ROBOT_CFG.spawn.articulation_props,
    ),
    init_state=_BASE_ROBOT_CFG.init_state.replace(
        joint_pos={
            **_BASE_ROBOT_CFG.init_state.joint_pos,
            "piperL_gripper": 0.1,
            "piperL_gripper_joint1": 0.05,
            "piperL_gripper_joint2": -0.05,
        },
    ),
    actuators={
        **_BASE_ROBOT_CFG.actuators,
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=ROBOT_GRIPPER_ACTUATOR_JOINT_NAMES,
            effort_limit=30.0,
            velocity_limit=3.0,
            stiffness=120.0,
            damping=5.0,
            friction=0.05,
            armature=0.0,
            min_delay=0,
            max_delay=1,
        ),
    },
)

OBJECT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    spawn=sim_utils.CuboidCfg(
        size=(0.05, 0.05, 0.08),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.18, 0.08)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.24)),
)

PICK_SUPPORT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/PickSupport",
    spawn=sim_utils.CylinderCfg(
        radius=0.18,
        height=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=False,
            linear_damping=50.0,
            angular_damping=50.0,
            max_linear_velocity=0.05,
            max_angular_velocity=1.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.42, 0.42, 0.38)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.10)),
)

PLACE_SUPPORT_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/PlaceSupport",
    spawn=sim_utils.CylinderCfg(
        radius=0.12,
        height=0.3,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            kinematic_enabled=False,
            linear_damping=50.0,
            angular_damping=50.0,
            max_linear_velocity=0.05,
            max_angular_velocity=1.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=10000.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.36, 0.42)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 0.15)),
)


@configclass
class ConfigSummary:
    class general:
        decimation = 8
        episode_length_s = 18.0
        render_interval = 8
        is_finite_horizon = True

    class sim:
        dt = 0.0025

    class env:
        num_envs = 4000
        num_actions = 5 + len(ROBOT_ARM_JOINT_NAMES) + 1
        num_actor_obs = 3 + 3 + 12 + 12 + 6 + 6 + 3 + 7 + 3 + 3 + 3 + 3 + 7 + num_actions
        num_critic_obs = num_actor_obs + 2
        action_history_length = 2
        clip_actions = 1.0
        clip_obs = 100.0
        RL_SIM_ENV_CONFIG_SUMMARY_DIR = RL_SIM_ENV_CONFIG_SUMMARY_DIR

        policy_type = {
            "actor_critic_type": "ActorCriticEncoder",
            "vae_type": "VAEBlind",
        }
        training_type = "rl"
        module_cfg_dict = {
            "amp": {"hidden_dims": [512, 256]},
            "vae": {
                "encoder_in_dim": 1,
                "encoder_hidden_dims": [16],
                "encoder_out_dim": 1,
                "encoder_head_dim_dict": {"obs_vel": 0, "obs_com": 0, "obs_mass": 0, "obs_latent": 1},
                "decoder_in_dim": 1,
                "decoder_hidden_dims": [16],
                "decoder_out_dim": 1,
                "activation": "elu",
            },
            "actor_critic": {
                "actor": {
                    "num_actor_obs": num_actor_obs,
                    "num_actions": num_actions,
                    "actor_hidden_dims": [512, 256, 128],
                    "actor_obs_normalization": False,
                },
                "privileged_encoder": {
                    "num_privileged_obs": 1,
                    "privileged_encoder_hidden_dims": [16],
                    "num_privileged_encoder_out": 1,
                },
                "heightmap_encoder": {
                    "num_heightmap_obs": 1,
                    "heightmap_encoder_hidden_dims": [16],
                    "num_heightmap_encoder_out": 1,
                },
                "critic": {
                    "num_critic_obs": num_critic_obs + 2,
                    "critic_hidden_dims": [512, 256, 128],
                    "critic_obs_normalization": False,
                },
                "init_noise_std": 0.35,
                "noise_std_type": "scalar",
                "activation": "elu",
                "min_normalized_std": [0.02] * num_actions,
                "max_normalized_std": [0.6, 0.6, 0.6, 0.6, 0.25] + [0.6] * (len(ROBOT_ARM_JOINT_NAMES) + 1),
            },
        }
        train_cfg_dict = {
            "use_amp": False,
            "use_vae": False,
            "ppo_algorithm": {
                "value_loss_coef": 0.5,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.0035,
                "entropy_coef_decay": 0.99995,
                "entropy_coef_min": 0.001,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": 3.0e-5,
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
                "normalize_advantage_per_mini_batch": False,
            },
        }

    class control:
        low_level_policy_path = os.path.abspath(
            os.path.join(
                RL_SIM_ENV_CONFIG_SUMMARY_DIR,
                "../../../../../../../../scripts/logs/locomotion/grq20_v2d4_piperL_loco_manip_lower/2026-06-29_16-13-14/model_30000.pt",
            )
        )
        low_level_policy_format = "checkpoint"
        base_command_low = (-0.45, -0.25, -0.5, 0.38, -0.15)
        base_command_high = (0.8, 0.25, 0.5, 0.47, 0.28)
        command_smoothing_alpha = 0.35
        arm_target_smoothing_alpha = 0.35
        arm_target_max_delta = 0.08
        arm_action_scale = (0.8, 0.8, 0.8, 0.8, 0.6, 0.6)
        gripper_action_scale = 0.08
        gripper_binary_action = True
        gripper_binary_threshold = 0.0
        gripper_closed_target = 0.0
        gripper_open_target = 0.1
        low_level_obs_dim = 47
        low_level_history_length = 5
        low_level_action_scale = 0.25
        low_level_decimation = 8
        low_level_onnx_use_cuda = True
        low_level_base_ang_vel_scale = 0.25
        low_level_joint_vel_scale = 0.05
        low_level_last_action_scale = 0.25
        low_level_command_scale = (2.0, 2.0, 0.25, 2.0, 1.0)
        low_level_actor_cfg = {
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
        low_level_vae_cfg = {
            "encoder_in_dim": 235,
            "encoder_hidden_dims": [128],
            "encoder_out_dim": 64,
            "encoder_head_dim_dict": {"obs_vel": 3, "obs_com": 3, "obs_mass": 1, "obs_latent": 16},
            "decoder_in_dim": 23,
            "decoder_hidden_dims": [64, 128],
            "decoder_out_dim": 51,
            "activation": "elu",
        }

    class task:
        use_dynamic_object = True
        phase_transition_distance = 0.14
        phase_transition_place_distance = 0.18
        phase2_hold_time_s = 0.3
        phase2_ee_place_threshold = 0.12
        phase2_base_place_threshold = 0.45
        phase2_heading_threshold = 0.5
        phase2_roll_limit = 0.4
        phase2_pitch_limit = 0.5
        pick_place_num_phases = 7
        post_place_hold_time_s = 0.2
        post_place_base_velocity_threshold = 0.08
        post_place_yaw_velocity_threshold = 0.25
        base_contact_termination_enable = False
        base_contact_force_threshold = 50.0
        base_contact_penalty_threshold = 10.0
        object_fallen_min_height = 0.12
        object_fallen_termination_phases = (0, 1)
        min_pick_place_distance = 0.65
        pick_place_curriculum_enable = True
        pick_place_curriculum_mode = "success"
        pick_place_curriculum_max_level = 10.0
        pick_place_curriculum_success_phase = 6
        pick_place_curriculum_failure_phase = 2
        pick_place_curriculum_level_increment = 0.25
        pick_place_curriculum_level_decrement = 0.25
        pick_place_curriculum_episode_mult = 30.0
        pick_place_curriculum_growth_power = 1.0
        pick_place_curriculum_steps_per_iteration = 24.0
        pick_place_curriculum_mid_iteration = 6000.0
        pick_place_curriculum_mid_progress = 1.0 / 20.0
        pick_place_curriculum_final_iteration = 12000.0
        pick_place_curriculum_slow_power = 1.5
        pick_place_curriculum_fast_power = 0.7
        pick_x_range_start = (0.95, 1.15)
        pick_y_range_start = (-0.25, 0.25)
        pick_z_range_start = (0.24, 0.24)
        place_x_range_start = (1.55, 1.85)
        place_y_range_start = (-0.4, 0.4)
        place_z_range_start = (0.34, 0.34)
        pick_x_range_final = (1.0, 5.0)
        pick_y_range_final = (-2.0, 2.0)
        pick_z_range_final = (0.03, 0.6)
        place_x_range_final = (1.0, 5.0)
        place_y_range_final = (-2.0, 2.0)
        place_z_range_final = (0.03, 0.8)

    class action:
        base_command_half_range = (0.8, 0.4, 0.8, 0.1, 0.25)
        scale = (0.8, 0.4, 0.8, 0.1, 0.25, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0)

    class observation:
        enable_actor_obs_noise = True
        leg_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_JOINT_NAMES)
        arm_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_ARM_JOINT_NAMES)
        ee_cfg = SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME)

        obs_term_dict = {
            "ground_truth_obs": {
                "base_ang_vel_gt": {"scale": 0.25},
                "projected_gravity_gt": {},
                "joint_pos_rel_gt": {"params": {"asset_cfg": leg_joint_cfg}},
                "joint_vel_rel_gt": {"scale": 0.05, "params": {"asset_cfg": leg_joint_cfg}},
                "arm_joint_pos_rel_gt": {"params": {"asset_cfg": arm_joint_cfg}},
                "arm_joint_vel_rel_gt": {"scale": 0.05, "params": {"asset_cfg": arm_joint_cfg}},
                "base_pitch_roll_height_gt": {},
                "ee_pose_b_gt": {"params": {"ee_asset_cfg": ee_cfg}},
                "object_pos_b_gt": {},
                "object_vel_b_gt": {"scale": 0.5},
                "pick_pos_b_gt": {},
                "place_pos_b_gt": {},
                "phase_gt": {"params": {"num_phases": 7}},
                "high_action_gt": {},
                "high_base_command_gt": {},
                "low_action_gt": {},
                "gt_padding_1": {},
                "gt_padding_2": {},
            },
            "amp_obs": {"amp_padding_1": {}},
            "noise_and_delay_obs": {
                "base_ang_vel_noisy": {"scale": 0.25, "noise": 0.05},
                "projected_gravity_noisy": {"noise": 0.02},
                "joint_pos_rel_noisy": {"noise": 0.01, "params": {"asset_cfg": leg_joint_cfg}},
                "joint_vel_rel_noisy": {"scale": 0.05, "noise": 0.3, "params": {"asset_cfg": leg_joint_cfg}},
                "arm_joint_pos_rel_noisy": {"noise": 0.015, "params": {"asset_cfg": arm_joint_cfg}},
                "arm_joint_vel_rel_noisy": {"scale": 0.05, "noise": 0.4, "params": {"asset_cfg": arm_joint_cfg}},
                "base_pitch_roll_height_noisy": {"noise": 0.01},
                "ee_pose_b_noisy": {"noise": 0.005, "params": {"ee_asset_cfg": ee_cfg}},
                "object_pos_b_noisy": {"noise": 0.01},
                "object_vel_b_noisy": {"scale": 0.5, "noise": 0.1},
                "pick_pos_b_noisy": {"noise": 0.005},
                "place_pos_b_noisy": {"noise": 0.005},
                "nad_padding_1": {},
            },
        }
        policy_obs_dict = {
            "actor_obs": {
                "terms": [
                    "base_ang_vel_noisy",
                    "projected_gravity_noisy",
                    "joint_pos_rel_noisy",
                    "joint_vel_rel_noisy",
                    "arm_joint_pos_rel_noisy",
                    "arm_joint_vel_rel_noisy",
                    "base_pitch_roll_height_noisy",
                    "ee_pose_b_noisy",
                    "object_pos_b_noisy",
                    "object_vel_b_noisy",
                    "pick_pos_b_noisy",
                    "place_pos_b_noisy",
                    "phase_gt",
                    "high_action_gt",
                ],
            },
            "critic_obs": {
                "terms": [
                    "base_ang_vel_gt",
                    "projected_gravity_gt",
                    "joint_pos_rel_gt",
                    "joint_vel_rel_gt",
                    "arm_joint_pos_rel_gt",
                    "arm_joint_vel_rel_gt",
                    "base_pitch_roll_height_gt",
                    "ee_pose_b_gt",
                    "object_pos_b_gt",
                    "object_vel_b_gt",
                    "pick_pos_b_gt",
                    "place_pos_b_gt",
                    "phase_gt",
                    "high_action_gt",
                    "gt_padding_1",
                    "gt_padding_2",
                ],
            },
            "low_actor_obs": {
                "terms": [
                    "base_ang_vel_gt",
                    "projected_gravity_gt",
                    "joint_pos_rel_gt",
                    "joint_vel_rel_gt",
                    "low_action_gt",
                    "high_base_command_gt",
                ],
            },
            "privileged_obs": {"terms": ["gt_padding_1"]},
            "gt_heightmap_obs": {"terms": ["gt_padding_2"]},
            "amp_obs": {"terms": ["amp_padding_1"]},
        }
        extra_obs_dict = {}

    class event:
        config_dict = {
            "reset_base": {
                "mode": "reset",
                "params": {
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.4, 0.4)},
                    "velocity_range": {k: (0.0, 0.0) for k in ["x", "y", "z", "roll", "pitch", "yaw"]},
                    "asset_cfg": SceneEntityCfg("robot"),
                    "object_cfg": SceneEntityCfg("object"),
                    "pick_support_cfg": SceneEntityCfg("pick_support"),
                    "place_support_cfg": SceneEntityCfg("place_support"),
                    "pick_x_range": (0.95, 1.15),
                    "pick_y_range": (-0.25, 0.25),
                    "pick_z_range": (0.24, 0.24),
                    "place_x_range": (1.55, 1.85),
                    "place_y_range": (-0.4, 0.4),
                    "place_z_range": (0.34, 0.34),
                    "min_pick_place_distance": 0.65,
                    "reference_asset_cfg": SceneEntityCfg("robot"),
                    "targets_in_reference_yaw_frame": True,
                    "pick_support_height": 0.2,
                    "place_support_height": 0.3,
                    "object_height": 0.08,
                    "object_xy_noise": (-0.01, 0.01),
                    "debug_vis": False,
                    "debug_vis_max_envs": 128,
                    "print_debug_once": True,
                },
            },
            "reset_robot_joints": {
                "mode": "reset",
                "params": {
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=ROBOT_JOINT_NAMES + ROBOT_ARM_JOINT_NAMES + ROBOT_GRIPPER_ACTUATOR_JOINT_NAMES,
                    ),
                    "position_range": (0.0, 0.0),
                    "velocity_range": (0.0, 0.0),
                },
            },
            "update_pick_place_phase": {
                "mode": "interval",
                "interval_range_s": (0.02, 0.02),
                "params": {
                    "ee_asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME),
                    "gripper_asset_cfg": SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES),
                    "object_cfg": SceneEntityCfg("object"),
                    "ee_object_threshold": 0.26,
                    "grasp_lift_height": 0.02,
                    "grasp_distance_threshold": 0.14,
                    "grasp_velocity_threshold": 0.8,
                    "grasp_object_speed_threshold": 0.8,
                    "gripper_closed_threshold": 0.06,
                    "phase1_min_grasp_time_s": 0.08,
                    "phase1_require_stable_grasp": False,
                    "phase2_min_hold_time_s": 0.12,
                    "phase2_require_grasp_to_advance": True,
                    "object_place_threshold": 10.0,
                    "object_place_settle_threshold": 0.25,
                    "gripper_open_threshold": 0.06,
                    "object_velocity_threshold": 0.35,
                    "require_object_still_for_phase4": False,
                    "require_place_height_for_release": True,
                    "place_height_threshold": 0.05,
                    "require_base_still_for_phase6": False,
                    "require_ee_clear_for_phase6": False,
                    "phase6_ee_object_min_distance": 0.12,
                    "post_place_hold_time_s": 0.2,
                    "post_place_base_velocity_threshold": 0.08,
                    "post_place_yaw_velocity_threshold": 0.25,
                },
            },
            "visualize_pick_place_targets": {
                "mode": "interval",
                "interval_range_s": (0.1, 0.1),
                "params": {
                    "debug_vis_max_envs": 128,
                    "debug_vis_pick_radius": 0.06,
                    "debug_vis_place_radius": 0.07,
                },
            },
        }

    class reward:
        only_positive_reward = False
        arm_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_ARM_JOINT_NAMES)
        ee_cfg = SceneEntityCfg("robot", body_names=ROBOT_EE_BODY_NAME)
        gripper_joint_cfg = SceneEntityCfg("robot", joint_names=ROBOT_GRIPPER_JOINT_NAMES)
        config_dict = {
            "base_to_pick_stance": {"weight": 0.15, "params": {"std": 0.6, "phases": (0,1)}},
            "ee_to_object": {"weight": 1.2, "params": {"ee_asset_cfg": ee_cfg, "std": 0.15, "phases": (0,1)}},
            "ee_to_target_shaped": {"weight": 0.2, "params": {"ee_asset_cfg": ee_cfg, "k_fast": 2.0, "k_slow": 0.5, "phases": (0,1)}},
            "ee_to_pick_progress": {"weight": 6.0, "params": {"ee_asset_cfg": ee_cfg, "phases": (0,1)}},
            "pick_reached_success": {"weight": 1.0, "params": {"ee_asset_cfg": ee_cfg, "distance_threshold": 0.05, "phases": (0,1)}},
            "ee_to_place_shaped": {"weight": 0.6, "params": {"ee_asset_cfg": ee_cfg, "k_fast": 2.0, "k_slow": 0.5, "phases": (2,)}},
            "ee_to_place_progress": {"weight": 3.0, "params": {"ee_asset_cfg": ee_cfg, "phases": (2,)}},
            "ee_to_object_shaped": {"weight": 0.8, "params": {"ee_asset_cfg": ee_cfg, "k_fast": 2.0, "k_slow": 0.5, "phases": (0,)}},
            "ee_object_contact": {"weight": 1.0, "params": {"ee_asset_cfg": ee_cfg, "distance_threshold": 0.05, "phases": (0,1)}},
            "grasp_success": {
                "weight": 120.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "distance_threshold": 0.14,
                    "velocity_threshold": 0.8,
                    "lift_height": 0.02,
                    "gripper_closed_threshold": 0.06,
                    "one_shot": True,
                    "phases": (2,),
                },
            },
            "grasp_hold_success": {
                "weight": 24.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "distance_threshold": 0.14,
                    "velocity_threshold": 0.8,
                    "lift_height": 0.02,
                    "gripper_closed_threshold": 0.06,
                    "require_lift": True,
                    "phases": (2, 3),
                },
            },
            "grasp_hold_failure_penalty": {
                "weight": -0.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "distance_threshold": 0.14,
                    "velocity_threshold": 0.8,
                    "lift_height": 0.02,
                    "gripper_closed_threshold": 0.06,
                    "require_lift": True,
                    "phases": (2, 3),
                },
            },
            "grasping_success_shaped": {
                "weight": 3.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "distance_threshold": 0.14,
                    "velocity_threshold": 0.8,
                    "gripper_closed_threshold": 0.06,
                    "held_reward": 0.25,
                    "lifted_reward": 1.75,
                    "phases": (1, 2),
                },
            },
            "grasp_stable_progress": {
                "weight": 8.0,
                "params": {"stable_time_s": 0.08, "phases": (1,)},
            },
            "object_lift_progress": {
                "weight": 8.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "lift_height": 0.02,
                    "distance_threshold": 0.14,
                    "velocity_threshold": 0.8,
                    "gripper_closed_threshold": 0.06,
                    "require_grasp": False,
                    "phases": (1,),
                },
            },
            "gripper_close_near_object": {
                "weight": 1.5,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "distance_threshold": 0.14,
                    "closed_threshold": 0.06,
                    "phases": (1,),
                },
            },
            "phase_transition_bonus": {
                "weight": 30.0,
                "params": {"bonuses": (10.0, 70.0, 190.0, 260.0, 180.0, 160.0)},
            },
            "phase_progress": {
                "weight": 9.0,
                "params": {"max_phase": 6, "one_shot": False, "normalize": True},
            },
            "object_below_pick_penalty": {
                "weight": -4.0,
                "params": {"margin": 0.025, "scale": 0.08, "phases": (1, 2, 3)},
            },
            "object_fallen_penalty": {
                "weight": -4.0,
                "params": {"min_height": 0.12, "phases": (2, 3, 4, 5, 6)},
            },
            "base_heading_to_place": {"weight": 0.15, "params": {"phases": (3,)}},
            "object_to_place_shaped": {
                "weight": 6.0,
                "params": {"k_wide": 2.0, "k_narrow": 4.0, "wide_gain": 3.0, "narrow_gain": 20.0, "phases": (2, 3)},
            },
            "object_to_place_progress": {"weight": 28.0, "params": {"phases": (2,3), "negative_scale": 0.15}},
            "base_to_place": {"weight": 0.5, "params": {"min_distance": 0.3, "gain": 0.5, "phases": (2, 3)}},
            "place_success": {
                "weight": 18.0,
                "params": {
                    "distance_threshold": 0.25,
                    "velocity_threshold": 0.35,
                    "require_object_still": False,
                    "phases": (3,),
                },
            },
            "gripper_release": {
                "weight": 45.0,
                "params": {
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "place_distance_threshold": 0.25,
                    "object_velocity_threshold": 0.35,
                    "require_object_still_for_place": False,
                    "require_place_height": True,
                    "place_height_threshold": 0.05,
                    "phases": (4, 5),
                },
            },
            "object_on_place_height_success": {
                "weight": 16.0,
                "params": {
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "place_xy_threshold": 0.25,
                    "height_std": 0.04,
                    "require_gripper_open": False,
                    "phases": (4, 5),
                },
            },
            "object_below_place_penalty": {
                "weight": -24.0,
                "params": {
                    "place_xy_threshold": 0.35,
                    "margin": 0.015,
                    "scale": 0.08,
                    "phases": (3, 4, 5, 6),
                },
            },
            "gripper_hold_after_place_penalty": {
                "weight": -35.0,
                "params": {
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "place_distance_threshold": 0.25,
                    "object_velocity_threshold": 0.35,
                    "require_object_still_for_place": False,
                    "phases": (4, 5, 6),
                },
            },
            "gripper_hold_near_place_penalty": {
                "weight": -22.0,
                "params": {
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "place_xy_threshold": 0.35,
                    "max_height_error": 0.25,
                    "phases": (3, 4, 5, 6),
                },
            },
            "object_hover_over_place_penalty": {
                "weight": -35.0,
                "params": {
                    "place_distance_threshold": 0.25,
                    "hover_margin": 0.015,
                    "hover_scale": 0.08,
                    "phases": (4, 5, 6),
                },
            },
            "post_place_still_success": {
                "weight": 18.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "distance_threshold": 0.25,
                    "object_velocity_threshold": 0.35,
                    "require_place_height": True,
                    "place_height_threshold": 0.05,
                    "require_base_still": False,
                    "require_ee_clear": False,
                    "ee_object_min_distance": 0.12,
                    "base_velocity_threshold": 0.08,
                    "yaw_velocity_threshold": 0.25,
                    "phases": (5,),
                },
            },
            "ee_retreat": {
                "weight": 8.0,
                "params": {"ee_asset_cfg": ee_cfg, "phases": (6,)},
            },
            "complete_success": {
                "weight": 18.0,
                "params": {
                    "ee_asset_cfg": ee_cfg,
                    "gripper_asset_cfg": gripper_joint_cfg,
                    "open_threshold": 0.06,
                    "place_distance_threshold": 0.25,
                    "object_velocity_threshold": 0.35,
                    "require_object_still_for_place": True,
                    "retreat_bonus": 0.0,
                    "require_still": True,
                    "base_velocity_threshold": 0.08,
                    "yaw_velocity_threshold": 0.25,
                    "phases": (6,),
                },
            },
            "phase2_hold_success": {"weight": 4.0, "params": {"phases": (2,)}},
            "base_planar_velocity": {"weight": -2.0, "params": {"yaw_weight": 0.25, "phases": (6,)}},
            "velocity_command_norm": {"weight": -2.0, "params": {"phases": (6,)}},
            "base_height_pitch_command": {
                "weight": -10.0,
                "params": {
                    "target_height": 0.42,
                    "target_pitch": 0.0,
                    "height_weight": 8.0,
                    "pitch_weight": 4.0,
                    "phases": (6,),
                },
            },
            "post_place_arm_joint_vel": {"weight": -0.05, "params": {"asset_cfg": arm_joint_cfg, "phases": (6,)}},
            "arm_nominal_pose_final": {
                "weight": -8.0,
                "params": {"asset_cfg": arm_joint_cfg, "phases": (6,)},
            },
            "high_action_rate": {"weight": -0.12},
            "high_action_acc": {"weight": -0.06},
            "base_contact_force": {
                "weight": -0.0005,
                "params": {
                    "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ROBOT_BASE_LINK),
                    "threshold": 10.0,
                },
            },
            "base_orientation": {"weight": -0.5},
            "base_height_final": {
                "weight": -10.0,
                "params": {"target_height": 0.42, "phases": (6,)},
            },
            "base_orientation_final": {"weight": -8.0, "params": {"phases": (6,)}},
            "arm_joint_vel": {"weight": -0.04, "params": {"asset_cfg": arm_joint_cfg, "phases": (0, 1, 2)}},
            "arm_action_rate": {"weight": -0.05, "params": {"phases": (0, 1, 2)}},
            "arm_torque": {"weight": -0.0001, "params": {"asset_cfg": arm_joint_cfg}},
            "arm_nominal_pose": {"weight": -0.02, "params": {"asset_cfg": arm_joint_cfg, "phases": (0, 5, 6)}},
            "arm_joint_limit": {"weight": -0.3, "params": {"asset_cfg": arm_joint_cfg, "margin_ratio": 0.03}},
            "base_vertical_vel": {"weight": -0.5},
            "base_roll_pitch_ang_vel": {"weight": -0.02},
            "velocity_command_rate": {"weight": -0.08},
            "velocity_command_acc": {"weight": -0.04},
            "body_command_norm": {"weight": -0.2},
            "body_command_rate": {"weight": -0.5},
            "body_command_acc": {"weight": -0.05},
            "excessive_pitch": {"weight": -0.5, "params": {"threshold": 0.25}},
            "excessive_height_change": {"weight": -0.5, "params": {"nominal_height": 0.42, "threshold": 0.1}},
            "support_clearance": {
                "weight": -0.0,
                "params": {
                    "support_names": ("pick_support", "place_support"),
                    "min_distance": 0.32,
                    "phases": (3,),
                },
            },
            "object_stability": {"weight": -0.2},
            "object_tilt_near_place": {"weight": -2.0},
            "ee_object_relative_velocity": {"weight": -0.2, "params": {"ee_asset_cfg": ee_cfg}},
        }


@configclass
class LocomotionPiperLPickPlaceRunnerCfg(LocomotionOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "grq20_v2d4_piperL_pick_place_gripper_object"
    swanlab_project = "grq20_v2d4_piperL_pick_place_gripper_object"

    policy_type = ConfigSummary.env.policy_type
    training_type = ConfigSummary.env.training_type
    module_cfg_dict = copy.deepcopy(ConfigSummary.env.module_cfg_dict)
    train_cfg_dict = copy.deepcopy(ConfigSummary.env.train_cfg_dict)
    amp_loader_cfg = AMPDataCfg()
