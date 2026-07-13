import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .piper_hook_cfg import PIPER_HOOK_CFG

from . import mdp
import isaaclab.envs.mdp as mdp_std

##
# 预定义配置
##

import os
import door_env

DOORWAY_CENTER_XY = (0.6, 0.0)
DOORWAY_FORWARD_AXIS = (0.0, -1.0)

##
# 场景定义
##


@configclass
class DoorEnvSceneCfg(InteractiveSceneCfg):
    """门环境场景配置。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(600.0, 600.0),
            color=(0.78, 0.80, 0.84),
        ),
    )

    # 门
    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{os.path.dirname(door_env.__file__)}/assets/Door_description/usd/Door.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=3.0  # 增加去穿透速度
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,  # 禁用门内部的自碰撞
                solver_position_iteration_count=12,  # 增加位置迭代次数
                solver_velocity_iteration_count=4,   # 增加速度迭代次数
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={"door_joint": 0.0, "handle_joint": 0.0}
        ),
        actuators={
            "door_joint": ImplicitActuatorCfg(
                joint_names_expr=["door_joint"],
                stiffness=20.0,
                damping=4.0,
                friction = 6.0,
                dynamic_friction = 6.0,
                viscous_friction = 1.0,
                effort_limit_sim = 50.0,
                velocity_limit_sim = 1.0, 
            ),
            "handle_joint": ImplicitActuatorCfg(
                joint_names_expr=["handle_joint"],
                stiffness=3.0,
                damping=0.3,
                friction=0.03,
                viscous_friction=0.03,
                dynamic_friction=0.01,
                effort_limit_sim=10.0,
                velocity_limit_sim=4.0,
            )
        },
    )

    # Robot
    robot = PIPER_HOOK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.4, 1.2, 0.43)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)
    # 设置初始关节角度 - 参考 haply_teleoperation.py
    robot.init_state.joint_pos = {
        "FL_hip_joint": -0.05,
        "FL_thigh_joint": 0.75,
        "FL_calf_joint": -1.5,
        "FR_hip_joint": 0.05,
        "FR_thigh_joint": 0.75,
        "FR_calf_joint": -1.5,
        "RL_hip_joint": -0.05,
        "RL_thigh_joint": 0.75,
        "RL_calf_joint": -1.5,
        "RR_hip_joint": 0.05,
        "RR_thigh_joint": 0.75,
        "RR_calf_joint": -1.5,
        "link1_joint": 0.0,
        "link2_joint": 0.20,
        "link3_joint": 0.0,
        "link4_joint": 0.0,
        "link5_joint": 0.0,
        "link6_joint": 0.0,
    }
    robot.spawn.activate_contact_sensors = True  # 启用接触传感器
    # End Effector Frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper_grasp_center",
                name="gripper_grasp_center",
            ),
        ],
    )

    hook_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_hook",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/handle_1"],
    )

    body_door_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/door_1"],
    )

    leg_door_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(FL|FR|RL|RR)_(hip|thigh|calf|foot)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/door_1"],
    )

    body_door_frame_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/base_link"],
    )

    leg_door_frame_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(FL|FR|RL|RR)_(hip|thigh|calf|foot)",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/base_link"],
    )

    
    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP 设置
##



@configclass
class ActionsCfg:
    high_level_action = mdp.HighLevelDoorOpenActionCfg(
        asset_name="robot",
        leg_joint_names=[
            "FL_hip_joint",
            "FR_hip_joint",
            "RL_hip_joint",
            "RR_hip_joint",
            "FL_thigh_joint",
            "FR_thigh_joint",
            "RL_thigh_joint",
            "RR_thigh_joint",
            "FL_calf_joint",
            "FR_calf_joint",
            "RL_calf_joint",
            "RR_calf_joint",
        ],
        arm_joint_names=["link1_joint", "link2_joint", "link3_joint", "link4_joint", "link5_joint", "link6_joint"],
        arm_action_scale=(0.8, 0.8, 0.8, 0.8, 0.6, 0.6),
        arm_target_smoothing_alpha=0.35,
        arm_target_max_delta=0.08,
        use_default_arm_offset=True,
        use_stage2_action_scale=True,
        stage2_arm_scale=0.60,
        stage2_wrist_scale=0.35,
        nominal_joint_pos=(0.0, 0.20, -0.20, 0.0, 0.0, 0.0),
        joint_pos_min=(-2.6179938, 0.0, -2.9670597, -2.2165681, -1.5620696, -2.0943951),
        joint_pos_max=( 2.6179938, 3.1415926,  0.0,        2.2165681,  1.5620696,  2.0943951),
        effort_limit=(50.0, 50.0, 50.0, 50.0, 50.0, 50.0),
        velocity_limit=(5.0, 5.0, 5.5, 5.5, 5.0, 5.0),
        arm_stiffness=25.0,
        arm_damping=1.0,
        armature=(0.02, 0.02, 0.02, 0.01, 0.01, 0.01),
        default_body_height=0.43,
    )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        robot_arm_joint_pos = ObsTerm(
            func=mdp_std.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["link[1-6]_joint"])},
        )
        robot_arm_joint_vel = ObsTerm(
            func=mdp_std.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["link[1-6]_joint"])},
        )
        last_applied_arm_delta = ObsTerm(
            func=mdp.last_applied_arm_delta,
            params={"action_name": "high_level_action"},
        )
        arm_q_des_error = ObsTerm(
            func=mdp.arm_q_des_error,
            params={"action_name": "high_level_action"},
        )

        last_high_base_action = ObsTerm(func=mdp.last_high_base_action)
        last_arm_action = ObsTerm(func=mdp.last_arm_action)
        high_base_command = ObsTerm(func=mdp.high_base_command_3d)

        base_velocity_b = ObsTerm(
            func=mdp.base_velocity_b,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        projected_gravity_b = ObsTerm(
            func=mdp.projected_gravity_b,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        base_height = ObsTerm(
            func=mdp.base_height,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )
        base_to_doorway_center_b_xy = ObsTerm(
            func=mdp.base_to_doorway_center_b_xy,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "doorway_center_xy": DOORWAY_CENTER_XY,
            },
        )
        doorway_forward_axis_b_xy = ObsTerm(
            func=mdp.doorway_forward_axis_b_xy,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            },
        )
        ee_to_handle_target_b = ObsTerm(
            func=mdp.ee_to_handle_target_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "handle_offset_h": (-0.08, 0.04, 0.01),
                "ee_offset_pos": (0.0, 0.0, 0.0),
            },
        )
        # Dynamic exteroception appended after the existing policy schema.
        # These quantities can later be supplied by AprilTag perception.
        handle_target_position_b = ObsTerm(
            func=mdp.handle_target_point_b,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "handle_offset_h": (-0.08, 0.04, 0.01),
            },
        )
        door_panel_forward_axis_b_xy = ObsTerm(
            func=mdp.door_panel_forward_axis_b_xy,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "panel_cfg": SceneEntityCfg("door", body_names=["door_1"]),
                "panel_forward_axis": (0.0, -1.0, 0.0),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedStateCfg(ObsGroup):
        ee_pos_in_handle = ObsTerm(
            func=mdp.ee_pos_in_handle_frame,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "ee_offset_pos": (0.0, 0.0, 0.0),
            },
        )

        ee_quat_err_in_handle = ObsTerm(
            func=mdp.ee_quat_error_handle_frame,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "ee_offset_pos": (0.0, 0.0, 0.0),
            },
        )

        handle_joint_pos = ObsTerm(
            func=mdp_std.joint_pos,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["handle_joint"])},
        )
        handle_joint_vel = ObsTerm(
            func=mdp_std.joint_vel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["handle_joint"])},
        )

        door_joint_pos = ObsTerm(
            func=mdp_std.joint_pos,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"])},
        )
        door_joint_vel = ObsTerm(
            func=mdp_std.joint_vel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=["door_joint"])},
        )
        unlock_state = ObsTerm(func=mdp.door_unlock_state)
        stage_id = ObsTerm(func=mdp.door_stage_id)
        body_door_contact_force_norm = ObsTerm(
            func=mdp.body_door_contact_force_norm,
            params={"sensor_name": "body_door_contact", "force_ref": 50.0},
        )
        leg_door_contact_force_norm = ObsTerm(
            func=mdp.leg_door_contact_force_norm,
            params={"sensor_name": "leg_door_contact", "force_ref": 50.0},
        )
        body_door_frame_contact_force_norm = ObsTerm(
            func=mdp.body_door_frame_contact_force_norm,
            params={"sensor_name": "body_door_frame_contact", "force_ref": 50.0},
        )
        leg_door_frame_contact_force_norm = ObsTerm(
            func=mdp.leg_door_frame_contact_force_norm,
            params={"sensor_name": "leg_door_frame_contact", "force_ref": 50.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # policy_obs_noisy is injected as a zero-copy alias by the distillation
    # VecEnv wrapper while no noise model is enabled, avoiding a second pass
    # over all policy observation terms.
    policy_obs_clean: PolicyCfg = PolicyCfg()
    privileged_state: PrivilegedStateCfg = PrivilegedStateCfg()



@configclass
class EventCfg:
    # 重置机器人 base/root 到配置里的初始世界位姿
    reset_robot_root = EventTerm(
        func=mdp.reset_root_state_to_default,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # 重置机器人关节到标准姿态
    reset_robot_joints = EventTerm(
        func=mdp_std.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["link[1-6]_joint", "(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
            ),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_door_joints = EventTerm(
        func=mdp_std.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
    },
)

    # 暂时关闭 stage archive reset，只保留标准 episode reset。
    staged_reset = None


    # 锁门事件：根据门把手角度限制门关节范围
    door_mechanism = EventTerm(
        func=mdp.update_door_lock_hysteresis_delayed_release,
        mode="interval",
        interval_range_s=[0.01, 0.01],
        params={
            "handle_joint_name": "handle_joint",
            "door_joint_name": "door_joint",

            # 深压到这里并保持几步，才进入 unlocked latch
            "unlock_threshold": -0.30,
            "unlock_hold_steps": 4,

            # unlocked 后继续锁住 door_joint 的步数
            "release_delay_steps": 2,

            # 只有门和把手都基本回到关闭位置才重新锁门
            "relock_door_threshold": 0.05,
            "relock_handle_threshold": -0.03,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,

            # 锁住时门关节位置
            "lock_door_pos": 0.0,
        },
    )

    visualize_doorway_debug = None

_HOOK_ALIGN_PARAMS = {
    "hand_cfg": SceneEntityCfg("robot", body_names=["gripper_grasp_center"]),
    "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
    "hook_approach_axis_hand": (0.0, 1.0, 0.0),
    "hook_mouth_axis_hand": (1.0, 0.0, 0.0),
    "handle_approach_axis": 1,
    "expected_approach_sign": -1.0,
    "world_down_axis": (0.0, 0.0, -1.0),
    "approach_weight": 0.70,
    "mouth_down_weight": 0.30,
}

_HOOK_KEEP_PARAMS = {
    **_HOOK_ALIGN_PARAMS,
    "contact_sensor_name": "hook_contact",
    "distance_threshold": 0.12,
    "ee_offset_pos": (0.0, 0.0, 0.0),
    "handle_offset_h": (-0.08, 0.04, 0.01),
    "align_threshold": 0.30,
}

_HOOK_CONTACT_KEEP_PARAMS = {
    **_HOOK_KEEP_PARAMS,
    "contact_threshold": 0.25,
}

_STAGE0_EE_CFG = SceneEntityCfg("robot", body_names=["gripper_grasp_center"])
_STAGE0_HANDLE_CFG = SceneEntityCfg("door", body_names=["handle_1"])
_STAGE0_HANDLE_OFFSET = (-0.08, 0.04, 0.01)

_STAGE0_GRASP_QUALITY_PARAMS = {
    "ee_cfg": _STAGE0_EE_CFG,
    "handle_cfg": _STAGE0_HANDLE_CFG,
    "contact_sensor_name": "hook_contact",
    "handle_offset_h": _STAGE0_HANDLE_OFFSET,
    "distance_threshold": 0.10,
    "contact_threshold": 0.25,
    "align_threshold": 0.30,
    "hook_approach_axis_hand": (0.0, 1.0, 0.0),
    "hook_mouth_axis_hand": (1.0, 0.0, 0.0),
    "handle_approach_axis": 1,
    "expected_approach_sign": -1.0,
    "world_down_axis": (0.0, 0.0, -1.0),
    "approach_weight": 0.70,
    "mouth_down_weight": 0.30,
}

# Stage-0 terms are declared independently and then injected into the stage
# aggregator. This keeps each reward tunable without flattening its full
# configuration into stage_gated_door_reward.
_STAGE0_BASE_TO_PICK_STANCE = {
    "func": mdp.base_to_pick_stance,
    "weight": 0.15,
    "params": {
        "robot_cfg": SceneEntityCfg("robot"),
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
        "stance_offset_w": (-0.3, 0.3, 0.0),
        "std": 0.6,
    },
}
_STAGE0_EE_TO_OBJECT = {
    "func": mdp.ee_to_object,
    "weight": 1.2,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
        "std": 0.15,
    },
}
_STAGE0_EE_TO_TARGET_SHAPED = {
    "func": mdp.ee_to_target_shaped,
    "weight": 0.2,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
        "k_fast": 2.0,
        "k_slow": 0.5,
    },
}
_STAGE0_EE_TO_PICK_PROGRESS = {
    "func": mdp.ee_to_pick_progress,
    "weight": 6.0,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
    },
}
_STAGE0_PICK_REACHED_SUCCESS = {
    "func": mdp.pick_reached_success,
    "weight": 1.0,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
        "distance_threshold": 0.05,
    },
}
_STAGE0_EE_OBJECT_CONTACT = {
    "func": mdp.ee_object_contact,
    "weight": 1.0,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "contact_sensor_name": "hook_contact",
        "handle_offset_h": _STAGE0_HANDLE_OFFSET,
        "distance_threshold": 0.05,
        "contact_threshold": 0.25,
    },
}
_STAGE0_EE_TO_OBJECT_SHAPED = {
    "func": mdp.ee_to_object_shaped,
    "weight": 0.8,
    "params": {
        "ee_cfg": _STAGE0_EE_CFG,
        "handle_cfg": _STAGE0_HANDLE_CFG,
        "k_fast": 2.0,
        "k_slow": 0.5,
    },
}
_STAGE0_GRASPING_SUCCESS_SHAPED = {
    "func": mdp.grasping_success_shaped,
    "weight": 3.0,
    "params": _STAGE0_GRASP_QUALITY_PARAMS,
}
_STAGE0_GRASP_STABLE_PROGRESS = {
    "func": mdp.grasp_stable_progress,
    "weight": 8.0,
    "params": {**_STAGE0_GRASP_QUALITY_PARAMS, "stable_time_s": 0.08},
}

_STAGE0_REWARD_TERMS = {
    "base_to_pick_stance": _STAGE0_BASE_TO_PICK_STANCE,
    "ee_to_object": _STAGE0_EE_TO_OBJECT,
    "ee_to_target_shaped": _STAGE0_EE_TO_TARGET_SHAPED,
    "ee_to_pick_progress": _STAGE0_EE_TO_PICK_PROGRESS,
    "pick_reached_success": _STAGE0_PICK_REACHED_SUCCESS,
    "ee_object_contact": _STAGE0_EE_OBJECT_CONTACT,
    "ee_to_object_shaped": _STAGE0_EE_TO_OBJECT_SHAPED,
    "grasping_success_shaped": _STAGE0_GRASPING_SUCCESS_SHAPED,
    "grasp_stable_progress": _STAGE0_GRASP_STABLE_PROGRESS,
}

_GRASP_SUCCESS_LATCH_TERM = {
    "weight": 1.0,
    "params": {
        "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
        **_HOOK_KEEP_PARAMS,
        "distance_threshold": 0.10,
        "force_threshold": 0.25,
        # At 50 Hz, allow the 0.08 s stable-progress term to reach 1.0
        # before the following step latches Stage 1.
        "hold_steps": 5,
        "bonus": 10.0,
        "require_wrap": True,
        "archive_cap": 512,
        "relax_near_after_handle_pos": -0.05,
        "less_than": True,
    },
}


@configclass
class RewardsCfg:

    stage_gated_door_reward = RewTerm(
        func=mdp.stage_gated_door_reward,
        weight=1.0,
        params={
            "enable_stage_gated_reward": True,
            "stage0_only_reward": False,
            "pre_grasp_cap": 0.0,
            "stage0_reward_terms": _STAGE0_REWARD_TERMS,
            # The sparse bonus is forced to zero in gated mode; this term only
            # updates the grasp-success latch used by the stage masks.
            "grasp_success_term": _GRASP_SUCCESS_LATCH_TERM,

            "press_handle_weight": 2.0,
            "press_handle_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "less_than": True,
                "vel_deadzone": 0.01,
                "vel_scale": 0.05,
                "opposite_penalty": 0.2,
                "clip": 1.0,
                "pos_deadzone": 1.0e-4,
                "pos_scale": 2.0e-3,
                "use_vel_ema": True,
                "vel_ema_alpha": 0.25,
            },

            "keep_handle_after_press_weight": 2.0,
            "keep_handle_after_press_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "handle_start_pos": 0.0,
                "handle_threshold": -0.30,
                "activate_progress": 0.20,
                "use_unlock_success_latch": True,
                "door_closed_pos": 0.0,
                "door_open_sign": 1.0,
                "push_enter_open": 0.02,
                "door_open_threshold": 0.35,
                "max_keep_steps_after_unlock": 24,
                "keep_until_door_open": True,
                "hold_reward": 0.01,
                "progress_boost": 0.04,
                "release_event_penalty": 0.20,
                "lost_penalty": 0.02,
                "auto_open_penalty": 0.05,
            },

            "stall_after_grasp_weight": 0.5,
            "stall_after_grasp_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "stall_pos": -0.10,
                "pos_scale": 0.03,
                "penalty": 0.02,
                "recent_window_steps": 200,
                "grace_steps": 10,
                "less_than": True,
            },

            "stall_after_press_weight": 0.5,
            "stall_after_press_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                "enter_depth": 0.25,
                "exit_depth": 0.22,
                "grace_steps": 60,
                "ramp_steps": 60,
                "door_closed_pos": 0.0,
                "door_open_sign": 1.0,
                "door_progress_threshold": 0.01,
                "max_penalty": 0.2,
                "less_than": True,
            },

            "unlock_progress_weight": 4.5,
            "unlock_progress_params": {
                "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "require_grasp_success": True,
                "handle_start_pos": 0.0,
                "reward_stop_pos": -0.30,
                "delta_power": 1.2,
                "ema_alpha": 0.7,
                "deadzone": 5.0e-5,
                "backtrack_penalty": 0.01,
                "delta_gain": 1.2,
                "abs_power": 1.8,
                "abs_gain": 0.35,
                "hold_start_ratio": 0.35,
                "hold_power": 1.6,
                "hold_gain": 0.25,
                "clip": 2.0,
            },

            "unlock_transition_weight": 1.0,
            "unlock_transition_params": {"bonus": 10.0},

            "push_door_weight": 8.0,
            "push_door_params": {
                "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
                **_HOOK_CONTACT_KEEP_PARAMS,
                "require_unlock_success_latch": True,
                "require_gate": True,
                "distance_threshold": 0.14,
                "align_threshold": 0.20,
                "door_open_sign": 1.0,
                "door_closed_pos": 0.0,
                "door_open_target": 0.35,
                "delta_scale": 1.0,
                "abs_scale": 0.2,
                "ema_alpha": 0.25,
                "deadzone": 1.0e-4,
                "backtrack_penalty": 0.1,
                "clip": 1.0,
            },
        },
    )

    base_hold = RewTerm(
        func=mdp.base_hold_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "stage3_start_angle": 0.10,
            "stage4_start_angle": 0.70,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "cmd_penalty_scale": 0.05,
            "pos_penalty_scale": 0.25,
            "yaw_penalty_scale": 0.10,
            "cmd_deadzone": 0.03,
            "pos_deadzone": 0.03,
            "yaw_deadzone": 0.08,
        },
    )

    base_push_follow = RewTerm(
        func=mdp.base_push_follow_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            "stage3_start_angle": 0.10,
            "stage4_start_angle": 0.70,
            "door_reward_start_angle": 0.20,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "k_yaw": 2.0,
            "progress_vel_scale": 0.5,
        },
    )

    base_traverse = RewTerm(
        func=mdp.base_traverse_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            "hook_contact_sensor_name": "hook_contact",
            "stage3_start_angle": 0.10,
            "stage4_start_angle": 0.70,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "k_lat": 6.0,
            "k_yaw": 2.0,
            "progress_vel_scale": 0.5,
            "release_contact_threshold": 0.2,
            "release_near_doorway_distance": 0.35,
            "keep_opening_reward": 1.0,
        },
    )

    base_safety = RewTerm(
        func=mdp.base_safety_reward,
        weight=1.0,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "body_door_sensor_name": "body_door_contact",
            "leg_door_sensor_name": "leg_door_contact",
            "body_frame_sensor_name": "body_door_frame_contact",
            "leg_frame_sensor_name": "leg_door_frame_contact",
            "force_ref": 50.0,
            "default_height": 0.43,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "stage3_start_angle": 0.10,
            "stage4_start_angle": 0.70,
            "early_body_door_weight": 1.0,
            "early_leg_door_weight": 0.5,
            "early_body_frame_weight": 1.0,
            "early_leg_frame_weight": 0.5,
            "late_body_door_weight": 0.5,
            "late_leg_door_weight": 0.2,
            "late_body_frame_weight": 0.5,
            "late_leg_frame_weight": 0.2,
            "cmd_rate_weight": 0.05,
            "height_pitch_weight": 2.0,
        },
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp_std.time_out, time_out=True)

    base_traverse_success = DoneTerm(
        func=mdp.base_traverse_success,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_center_xy": DOORWAY_CENTER_XY,
            "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "min_door_angle": 0.70,
            "pass_distance": 0.5,
            "num_steps": 3,
        },
    )

    base_bad_orientation = DoneTerm(
        func=mdp.base_bad_orientation,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "doorway_forward_axis": DOORWAY_FORWARD_AXIS,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "traverse_stage_angle": 0.70,
            "min_base_displacement": 0.15,
            "require_traverse_stage": False,
            "require_base_displacement": False,
            "hard_yaw_error": 2.09,
            "hard_steps": 60,
            "soft_yaw_error": 1.57,
            "soft_steps": 100,
        },
    )

    base_fall = DoneTerm(
        func=mdp.base_fall,
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "min_base_height": 0.20,
        },
    )


##
# 环境配置
##


@configclass
class DoorEnvEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene: DoorEnvSceneCfg = DoorEnvSceneCfg(num_envs=4096, env_spacing=4.0)
    # 基础设置
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP 设置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # 初始化后处理
    def __post_init__(self) -> None:
        """初始化后处理。"""
        # 通用设置
        # 400 Hz physics / 50 Hz policy, matching the low-level locomotion checkpoint.
        self.decimation = 8
        self.episode_length_s = 15.0
        # 观察者设置
        self.viewer.eye = (7.0, -0.5, 4.0)
        # 仿真设置
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_max_rigid_patch_count = 2**19
