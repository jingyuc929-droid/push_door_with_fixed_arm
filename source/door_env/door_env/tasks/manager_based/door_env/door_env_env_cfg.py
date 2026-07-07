import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
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

from .arx5_cfg import ARX5_CFG

from . import mdp
import isaaclab.envs.mdp as mdp_std

##
# 预定义配置
##

import os
import door_env

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
            size=(100.0, 100.0),
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
                solver_position_iteration_count=20,  # 增加位置迭代次数
                solver_velocity_iteration_count=8,   # 增加速度迭代次数
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.002,  # 减小接触偏移
                rest_offset=0.0,       # 改为0,防止穿透
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
    robot = ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (1.00, 0.45, 0.75)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)
    # 设置初始关节角度 - 参考 haply_teleoperation.py
    robot.init_state.joint_pos = {
        "joint1": 0.0,
        "joint2": 0.20,
        "joint3": 0.20,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
        "gripper_joint": 0.00,
        "joint8": 0.00,
    }
    robot.spawn.activate_contact_sensors = True  # 启用接触传感器
    robot.spawn.collision_props = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.002,
        rest_offset=0.0,
    )


    # End Effector Frame
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link6",
                name="ee_tcp",
                offset=OffsetCfg(
                    pos=(0.1523,0.0,0.0),
                ),
            ),
        ],
    )

    #left_finger_contact_sensor
    left_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/handle_1"],
    )

    #right_finger_contact_sensor
    right_finger_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link8",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Door/handle_1"],
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
    arm_action = mdp.ARX5MITJointActionCfg(
        asset_name="robot",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        use_delta_mode=True,
        delta_scale=(0.015, 0.015, 0.015, 0.020, 0.020, 0.020),
        use_stage2_action_scale=True,
        stage2_arm_scale=0.60,
        stage2_wrist_scale=0.35,
        nominal_joint_pos=(0.0, 0.20, 0.20, 0.0, 0.0, 0.0),
        joint_pos_min=(-3.14, -0.05, -0.10, -1.60, -1.57, -2.00),
        joint_pos_max=( 2.618, 3.50,  3.20,  1.55,  1.57,  2.00),
        effort_limit=(30.0, 40.0, 30.0, 15.0, 10.0, 10.0),
        velocity_limit=(5.0, 5.0, 5.5, 5.5, 5.0, 5.0),
        kp=(80.0, 70.0, 70.0, 30.0, 30.0, 20.0),
        kd=(2.0, 2.0, 2.0, 1.0, 1.0, 0.7),
        armature=(0.02, 0.02, 0.02, 0.01, 0.01, 0.01),
    )

    gripper_action = mdp_std.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper_joint"],
        open_command_expr={"gripper_joint": 0.044},
        close_command_expr={"gripper_joint": -0.005},
    )

@configclass
class ObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        # robot proprio: 建议显式限制，不要把整个 articulation 全读进来
        robot_arm_joint_pos = ObsTerm(
            func=mdp_std.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
        )
        robot_arm_joint_vel = ObsTerm(
            func=mdp_std.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-6]"])},
        )

        last_action = ObsTerm(
            func=mdp.last_action,
            params={"action_names": ("arm_action", "gripper_action"), "action_dims": (6, 1)},
        )

        # Low-level controller state available to future deployable Student policies.
        last_applied_arm_delta = ObsTerm(
            func=mdp.last_applied_arm_delta,
            params={"action_name": "arm_action"},
        )
        arm_q_des_error = ObsTerm(
            func=mdp.arm_q_des_error,
            params={"action_name": "arm_action"},
        )

        gripper_opening = ObsTerm(
            func=mdp.gripper_opening,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        ee_tcp_pose_w = ObsTerm(
            func=mdp.ee_tcp_pose_w,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "ee_offset_pos": (0.1523, 0.0, 0.0),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        ee_pos_in_handle = ObsTerm(
            func=mdp.ee_pos_in_handle_frame,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "ee_offset_pos": (0.1523, 0.0, 0.0),
            },
        )

        ee_quat_err_in_handle = ObsTerm(
            func=mdp.ee_quat_error_handle_frame,
            params={
                "ee_cfg": SceneEntityCfg("robot", body_names=["link6"]),
                "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
                "ee_offset_pos": (0.1523, 0.0, 0.0),
            },
        )

        handle_pose = ObsTerm(
            func=mdp_std.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("door", body_names=["handle_1"])},
        )

        door_joint_pos = ObsTerm(
            func=mdp_std.joint_pos,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=[".*"])},
        )
        door_joint_vel = ObsTerm(
            func=mdp_std.joint_vel,
            params={"asset_cfg": SceneEntityCfg("door", joint_names=[".*"])},
        )

        finger_contact_norms = ObsTerm(
            func=mdp.finger_contact_norms,
            params={
                "left_sensor_name": "left_finger_contact",
                "right_sensor_name": "right_finger_contact",
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()



@configclass
class EventCfg:
    # 重置机器人关节到标准姿态
    reset_robot_joints = EventTerm(
        func=mdp_std.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["joint[1-6]", "gripper_joint", "joint8"],
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

    # 重置机器人和门到保存好的stage状态
    staged_reset = EventTerm(
       func=mdp.staged_reset_from_archive,
       mode="reset",
        params={
           "robot_cfg": SceneEntityCfg("robot"),
           "door_cfg": SceneEntityCfg("door"),
           "p_grasp_start": 0.85,
           "p_unlock_start": 0.10,   # 先攒 archive，所以可以先调小一点看看效果
           "min_archive": 32,
           "cap_grasp": 512,
           "cap_unlock": 512,
       },
    )


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


@configclass
class RewardsCfg:
    #align_grasp = RewTerm(
        #func=mdp.align_grasp_around_handle_local,
        #weight=4.5,
        #params={ 
        #    "left_finger_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger"]), 
        #    "right_finger_cfg": SceneEntityCfg("robot", body_names=["panda_rightfinger"]), 
        #    "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]), 
        #    "grasp_axis": 2, 
        #    "min_sep": 0.002,
        #}, 
#)

    align_grasp = RewTerm(
    func=mdp.align_grasp_pose_v2,
    weight=4.0,
    params={
        "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
        "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
        "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
        "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),

        # 1) side
        "grasp_axis": 2,
        "min_sep": 0.010,
        "sep_scale": 0.010,
        "symmetry_scale": 0.015,

        # 3) open-axis align
        "gripper_open_axis_hand": (0.0, 1.0, 0.0),

        # 4) approach-axis align
        "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
        "handle_approach_axis": 1,

        # combine weights
        "side_weight": 0.40,
        "open_weight": 0.20,
        "approach_weight": 0.40,
    },
)

    # 2) 强接近信号：ee_tcp -> 把手目标抓取点（inverse-square, 远处也有梯度）
    approach_handle = RewTerm(
        func=mdp.approach_handle_inv_square,
        weight=0.70,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "ee_offset_pos": (0.1523, 0.0, 0.0),
            "handle_offset_h":  (-0.08, 0.04, 0.01),
            "eps": 1.0e-4,
            "scale": 0.02,
            "clip": 5.0,
        },
    )

    # 3) 近 + 环绕对齐后，鼓励闭合（解决“鸡生蛋”：没闭合就很难双指接触）
    close_when_ready = RewTerm(
    func=mdp.close_gripper_shaping_when_ready,
    weight=3.5,
    params={
        "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
        "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
        "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
        "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
        "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),

        "distance_threshold": 0.05,
        "ee_offset_pos": (0.1523, 0.0, 0.0),
        "handle_offset_h": (-0.08, 0.04, 0.01),

        "grasp_axis": 2,
        "min_sep": 0.005,
        "sep_scale": 0.010,
        "symmetry_scale": 0.015,
        "gripper_open_axis_hand": (0.0, 1.0, 0.0),
        "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
        "handle_approach_axis": 1,
        "side_weight": 0.40,
        "open_weight": 0.20,
        "approach_weight": 0.40,
        "align_threshold": 0.30,

        "require_any_contact": False,
        "contact_threshold": 0.1,
        "open_width": 0.09,
    },
)

    # 4) 抓取奖励（方向性力：只奖 |Fz|，罚侧向力；handle-only filtered contact）
    grasp_handle = RewTerm(
    func=mdp.grasp_handle_reward_preunlock_only,
    weight=6.0,
    params={
        "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
        "unlock_enter_pos": -0.02,
        "fade_width": 0.2,
        "less_than": True,

        "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
        "handle_offset_h": (-0.08, 0.04, 0.01),
        "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
        "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
        "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
        "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
        "left_sensor_name": "left_finger_contact",
        "right_sensor_name": "right_finger_contact",

        "distance_threshold": 0.10,
        "grasp_axis": 2,
        "min_sep": 0.0050,
        "sep_scale": 0.010,
        "symmetry_scale": 0.015,
        "gripper_open_axis_hand": (0.0, 1.0, 0.0),
        "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
        "handle_approach_axis": 1,
        "align_side_weight": 0.40,
        "align_open_weight": 0.20,
        "align_approach_weight": 0.40,
        "align_threshold": 0.50,

        "force_threshold": 0.5,
        "force_scale": 10.0,
        "side_scale": 10.0,
        "side_weight": 0.3,

        "open_width": 0.09,
        "min_closedness": 0.3,
        "close_scale": 1.0,
        "close_power": 2.0,

        "hold_steps": 2,
        "hold_tau": 2.5,
        "hold_power": 1.0,
        "hold_decay": 0.6,
        "balance_power": 2.0,
        "finger_speed_std": 0.08, 
    },
)

    # 5) grasp成功一次性奖励 bonus = 20, 已经抓住把手后保持夹爪张开会有惩罚
    grasp_success = RewTerm(
    func=mdp.grasp_success_bonus,
    weight=1.0,
    params={
        "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
        "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
        "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
        "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
        "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
        "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
        "left_sensor_name": "left_finger_contact",
        "right_sensor_name": "right_finger_contact",

        "distance_threshold": 0.1,
        "require_any_finger_contact": False,
        "use_force_norm": True,
        "near_mode": "mid",
        "use_grasp_point": True,
        "handle_offset_h":  (-0.08, 0.04, 0.01),
        "archive_cap": 512,

        "hold_steps": 4,

        "grasp_axis": 2,
        "min_sep": 0.0050,   #最小的两指距离，单位为m        "sep_scale": 0.010,
        "sep_scale": 0.0010,
        "symmetry_scale": 0.015,
        "gripper_open_axis_hand": (0.0, 1.0, 0.0),
        "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
        "handle_approach_axis": 1,
        "align_side_weight": 0.70,
        "align_open_weight": 0.10,
        "align_approach_weight": 0.20,
        "align_threshold": 0.30,
        "bonus": 18.0,
    },
)
    

    keep_handle_after_press = RewTerm(
        func=mdp.anti_release_after_press_to_open,
        weight=4.0,
        params={
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",

            "contact_threshold": 0.3,
            "require_any_contact": False,
            "open_width": 0.09,
            "min_closedness": 0.40,
            "distance_threshold": 0.1,
            "handle_offset_h": (-0.08, 0.04, 0.01),

            "handle_start_pos": 0.0,
            "handle_threshold": -0.30,
            "activate_progress": 0.18,
            "use_unlock_success_latch": True,

            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "push_enter_open": 0.02,
            "door_open_threshold": 0.35,

            "max_keep_steps_after_unlock": 480,
            "keep_until_door_open": True,

            "hold_reward": 0.02,
            "progress_boost": 0.04,
            "release_event_penalty": 0.35,
            "lost_penalty": 0.25,
            "auto_open_penalty": 0.50,
        },
    )

    grasp_quality_keep = RewTerm(
        func=mdp.grasp_quality_keep_reward,
        weight=0.5,
        params={
            "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",
            "ee_offset_pos": (0.1523, 0.0, 0.0),
            "handle_offset_h": (-0.08, 0.04, 0.01),
            "near_sigma": 0.06,
            "near_hard_threshold": 0.14,
            "grasp_axis": 2,
            "min_sep": 0.005,
            "sep_scale": 0.010,
            "symmetry_scale": 0.015,
            "gripper_open_axis_hand": (0.0, 1.0, 0.0),
            "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
            "handle_approach_axis": 1,
            "expected_approach_sign": 1.0,
            "contact_threshold": 0.25,
            "contact_scale": 0.50,
            "balance_power": 0.5,
            "open_width": 0.09,
            "min_closedness": 0.35,
            "target_closedness": 0.65,
            "max_closedness": 0.98,
            "single_force_high": 1.0,
            "single_force_low": 0.15,
            "closed_no_contact_penalty": 0.5,
            "single_finger_penalty": 0.5,
        },
    )

    
    # 6）鼓励下压把手，防止局部最优

    press_after_grasp = RewTerm(
        func=mdp.press_handle_after_grasp_vel,
        weight=8.0,
        params={
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",
            "contact_threshold": 0.30,
            "require_any_contact": True,
            "open_width": 0.09,
            "min_closedness": 0.45,
            "less_than": True,
            "vel_deadzone": 0.002,
            "vel_scale": 0.03,
            "vel_ema_alpha":0.25,        
            "pos_deadzone": 0.0001,
            "pos_scale": 0.003,
            "opposite_penalty": 0.0,
            "clip": 1.0,
        },
)

    stall_after_grasp = RewTerm(
        func=mdp.stall_penalty_after_grasp_pos,
        weight=3.5,
        params={
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",
            "contact_threshold": 0.2,
            "require_any_contact": False,
            "stall_pos": -0.06,
            "pos_scale": 0.04,
            "penalty": 0.85,
            "recent_window_steps": 200,
            "grace_steps": 10,
            "less_than": True,
        },
    )   

    # 7) 解锁进度奖励：越往深处压奖励越大，只奖励刷新本 episode 最深解锁进度
    unlock_progress = RewTerm(
        func=mdp.unlock_handle_progress_mixed,
        weight=10.0,
        params={
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",

            # gate
            "contact_threshold": 0.5,
            "require_any_contact": False,
            "open_width": 0.09,
            "min_closedness": 0.50,
            "require_grasp_success": True,

            # progress definition
            "handle_start_pos": 0.0,
            "reward_stop_pos": -0.36,

            # delta branch
            "delta_power": 1.2,
            "ema_alpha": 0.7,
            "deadzone": 5e-5,
            "backtrack_penalty": 0.01,
            "delta_gain": 1.2,

            # absolute branch
            "abs_power": 1.8,
            "abs_gain": 0.35,

            # deep-hold branch
            "hold_start_ratio": 0.35,
            "hold_power": 1.6,
            "hold_gain": 0.25,

            # final clip
            "clip": 2.0,
        },
    )

    stall_after_press = RewTerm(
        func=mdp.near_unlock_stall_penalty,
        weight=1.0,
        params={
            "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "enter_depth": 0.26,
            "exit_depth": 0.22,
            "grace_steps": 60,
            "ramp_steps": 60,
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "door_progress_threshold": 0.02,
            "max_penalty": 1.0,
            "less_than": True,
        },
    )

    unlock_transition = RewTerm(
        func=mdp.physical_unlock_transition_bonus,
        weight=1.0,
        params={
            "bonus": 10.0,
        },
    )

    # 10) 门打开奖励（根据门的开度线性给奖励，鼓励持续开门）
    push_door = RewTerm(
    func=mdp.push_door_progress_after_unlock_success_only,
    weight=11.0,
    params={
        "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
        "require_unlock_success_latch": True,
        "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
        "left_sensor_name": "left_finger_contact",
        "right_sensor_name": "right_finger_contact",
        "contact_threshold": 0.25,
        "require_any_contact": False,
        "open_width": 0.09,
        "min_closedness": 0.35,
        "require_gate": True,

        "door_open_sign": 1.0,
        "door_closed_pos": 0.0,
        "door_open_target": 0.35,

        "delta_scale": 1.0,
        "abs_scale": 0.2,
        "ema_alpha": 0.25,
        "deadzone": 1e-4,
        "backtrack_penalty": 0.1,
        "clip": 1.0,
    },
)

    stage_gated_door_reward = RewTerm(
        func=mdp.stage_gated_door_reward,
        weight=1.0,
        params={
            "enable_stage_gated_reward": True,
            # Stage0-only warmup: train approach + stable grasp first.
            # Set this back to False to resume stage1/stage2 unlock/open rewards.
            "stage0_only_reward": False,
            "pre_grasp_cap": 0.0,

            "align_grasp_weight": align_grasp.weight,
            "align_grasp_params": align_grasp.params,
            "approach_handle_weight": approach_handle.weight,
            "approach_handle_params": approach_handle.params,
            "close_when_ready_weight": close_when_ready.weight,
            "close_when_ready_params": close_when_ready.params,

            "grasp_handle_weight": grasp_handle.weight,
            "grasp_handle_params": grasp_handle.params,
            "grasp_success_weight": grasp_success.weight,
            "grasp_success_params": grasp_success.params,

            "press_handle_weight": press_after_grasp.weight,
            "press_handle_params": press_after_grasp.params,
            "keep_handle_after_press_weight": keep_handle_after_press.weight,
            "keep_handle_after_press_params": keep_handle_after_press.params,
            "grasp_quality_keep_weight": 0.5,
            "grasp_quality_keep_params": grasp_quality_keep.params,
            "grasp_quality_gate_floor": 0.15,
            "stall_after_grasp_weight": stall_after_grasp.weight,
            "stall_after_grasp_params": stall_after_grasp.params,
            "stall_after_press_weight": stall_after_press.weight,
            "stall_after_press_params": stall_after_press.params,
            "unlock_progress_weight": unlock_progress.weight,
            "unlock_progress_params": unlock_progress.params,

            "unlock_transition_weight": unlock_transition.weight,
            "unlock_transition_params": unlock_transition.params,
            "push_door_weight": push_door.weight,
            "push_door_params": push_door.params,

        },
    )

    # The aggregate term above owns reward activation. Keep the legacy terms
    # configured for rollback/reference, but prevent double-counting.
    align_grasp.weight = 0.0
    approach_handle.weight = 0.0
    close_when_ready.weight = 0.0
    grasp_handle.weight = 0.0
    grasp_success.weight = 0.0
    keep_handle_after_press.weight = 0.0
    grasp_quality_keep.weight = 0.0
    press_after_grasp.weight = 0.0
    stall_after_grasp.weight = 0.0
    stall_after_press.weight = 0.0
    unlock_progress.weight = 0.0
    unlock_transition.weight = 0.0
    push_door.weight = 0.0


@configclass
class TerminationsCfg:
    # (1) 超时终止（truncation）
    time_out = DoneTerm(func=mdp_std.time_out, time_out=True)

    # (2) 打开门（termination）
    door_open_success = DoneTerm(
        func=mdp.door_open_success_only,
        params={
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "door_open_threshold": 0.32,
            "num_steps": 3,
        },
    )

    # release_after_grasp_failure = DoneTerm(
    #     func=mdp.release_after_grasp_failure,
    #     params={
    #         "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
    #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
    #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
    #         "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
    #         "left_sensor_name": "left_finger_contact",
    #         "right_sensor_name": "right_finger_contact",
    #         "contact_threshold": 0.20,
    #         "require_any_contact": False,
    #         "open_width": 0.09,
    #         "min_closedness": 0.30,
    #         "distance_threshold": 0.15,
    #         "handle_offset_h": (-0.08, 0.04, 0.01),
    #         "grace_steps": 15,
    #         "fail_steps": 90,
    #     },
    # )

    release_after_unlock_failure = DoneTerm(
        func=mdp.release_after_unlock_failure,
        params={
            "door_joint_cfg": SceneEntityCfg("door", joint_names=["door_joint"]),
            "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
            "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
            "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
            "left_sensor_name": "left_finger_contact",
            "right_sensor_name": "right_finger_contact",
            "contact_threshold": 0.25,
            "require_any_contact": False,
            "open_width": 0.09,
            "min_closedness": 0.35,
            "distance_threshold": 0.13,
            "handle_offset_h": (-0.08, 0.04, 0.01),
            "door_closed_pos": 0.0,
            "door_open_sign": 1.0,
            "door_open_threshold": 0.35,
            "fail_steps": 30,
            "after_unlock_grace_steps": 10,
        },
    )
    

    
    # unlock_success = DoneTerm(
    #     func=mdp.unlock_handle_after_grasp,
    #     params={
    #         # --- gate: contact + closure (+ optional near) ---
    #         "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
    #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
    #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
    #         "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),
    #         "left_sensor_cfg": SceneEntityCfg("left_finger_contact"),
    #         "right_sensor_cfg": SceneEntityCfg("right_finger_contact"),

    #         "num_steps": 8,
    #         "force_threshold": 1.0,
    #         "require_any_contact": False,
    #         "require_near": True,
    #         "distance_threshold": 0.12,
    #         "open_width": 0.09,
    #         "min_closedness": 0.5,

    #         # --- unlock check ---
    #         "handle_joint_cfg": SceneEntityCfg("door", joint_names=["handle_joint"]),
    #         "handle_threshold": -0.3,
    #         "less_than": True,
    #     },
    # )


    #  抓住把手成功（termination）
    # - handle-only filtered contact（排除自碰/门板/地面）
    # - 连续 num_steps 满足
    # - 近距离 + 环绕对齐 + 一定闭合程度
    # grasp_sustained = DoneTerm(
    #     func=mdp.grasp_handle_sustained,
    #     params={
    #         # geometry
    #         "handle_cfg": SceneEntityCfg("door", body_names=["handle_1"]),
    #         "left_finger_cfg": SceneEntityCfg("robot", body_names=["link7"]),
    #         "right_finger_cfg": SceneEntityCfg("robot", body_names=["link8"]),
    #         "hand_cfg": SceneEntityCfg("robot", body_names=["link6"]),
    #         "gripper_cfg": SceneEntityCfg("robot", joint_names=["gripper_joint"]),

    #         # sensors (keys in env.scene)
    #         "left_sensor_cfg": SceneEntityCfg("left_finger_contact"),
    #         "right_sensor_cfg": SceneEntityCfg("right_finger_contact"),

    #         # sustained contact
    #         "num_steps": 5,
    #         "force_threshold": 1.0,
    #         "require_any_finger_contact": False,
    #         "use_force_norm": True,

    #         # near gate
    #         "distance_threshold": 0.10,
    #         "use_grasp_point": True,
    #         "handle_offset_h": (-0.08, 0.04, 0.01),
    #         # wrap gate
    #         "require_wrap": True,
    #         "grasp_axis": 2,
    #         "min_sep": 0.010,
    #         "sep_scale": 0.010,
    #         "symmetry_scale": 0.015,
    #         "gripper_open_axis_hand": (0.0, 1.0, 0.0),
    #         "gripper_approach_axis_hand": (1.0, 0.0, 0.0),
    #         "handle_approach_axis": 1,
    #         "align_side_weight": 0.70,
    #         "align_open_weight": 0.10,
    #         "align_approach_weight": 0.20,
    #         "align_threshold": 0.30,

    #         # closure gate
    #         "open_width": 0.09,
    #         "min_closedness": 0.5,
    #     },
    # )


##
# 环境配置
##


@configclass
class DoorEnvEnvCfg(ManagerBasedRLEnvCfg):
    # 场景设置
    scene: DoorEnvSceneCfg = DoorEnvSceneCfg(num_envs=1, env_spacing=4.0)
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
        self.decimation = 2
        self.episode_length_s = 12.0
        # 观察者设置
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 仿真设置
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
