import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


PIPER_HOOK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jing/push_door/source/door_env/door_env/assets/grq20_v2d4_piperL_front_mount_gripper_hook/grq20_v2d4_piperL_front_mount_gripper_hook.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
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
            "link3_joint": -0.20,
            "link4_joint": 0.0,
            "link5_joint": 0.0,
            "link6_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["(FL|FR|RL|RR)_(hip|thigh|calf)_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=20.94,
            stiffness=120.0,
            damping=8.0,
        ),
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["link[1-6]_joint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=6.0,
            stiffness=25.0,
            damping=1.0,
        ),
    },
)
