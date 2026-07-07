import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


ARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/jing/push_door/source/door_env/door_env/assets/X5/X5.usd",
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
            fix_root_link=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.002,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.20,
            "joint3": 0.20,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint": 0.02,
            "joint8": 0.02,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=120.0,
            velocity_limit_sim=6.0,
            stiffness=0.0,
            damping=0.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=200.0,
            damping=20.0,
        ),
    },
)