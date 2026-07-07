import argparse

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Standalone ARX5 USD debug launcher.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/home/jing/push_door/source/robot_env/X5/X5.usd",
    help="Absolute path to the ARX5 USD file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Isaac Lab imports (after AppLauncher)
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass


# -----------------------------------------------------------------------------
# Robot config
# -----------------------------------------------------------------------------
ARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=args_cli.usd_path,
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
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.2,
            "joint3": 0.2,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper_joint": 0.02,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=120.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=8.0,
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


@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    num_envs = args_cli.num_envs
    env_spacing = 2.0
    replicate_physics = False
    robot = ARX5_CFG.replace(prim_path="/World/envs/env_.*/Robot")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0, use_fabric=not args_cli.disable_fabric)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.5])

    # world assets
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    scene_cfg = DebugSceneCfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot: Articulation = scene["robot"]

    print("\n========== ROBOT BASIC INFO ==========")
    print("usd_path      =", args_cli.usd_path)
    print("is_fixed_base =", robot.is_fixed_base)
    print("num_joints    =", robot.num_joints)
    print("num_bodies    =", robot.num_bodies)

    print("\n========== JOINT NAMES ==========")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:02d}: {name}")

    print("\n========== BODY NAMES ==========")
    for i, name in enumerate(robot.body_names):
        print(f"{i:02d}: {name}")

    arm_cfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"], preserve_order=True)
    arm_cfg.resolve(scene)
    print("\n========== ARM JOINT RESOLVE ==========")
    print("joint_ids   =", arm_cfg.joint_ids)
    print("joint_names =", arm_cfg.joint_names)

    grip_cfg = SceneEntityCfg("robot", joint_names=["gripper_joint"])
    grip_cfg.resolve(scene)
    print("\n========== GRIPPER JOINT RESOLVE ==========")
    print("joint_ids   =", grip_cfg.joint_ids)
    print("joint_names =", grip_cfg.joint_names)

    candidate_bodies = [
        "gripper_center",
        "left_pad",
        "right_pad",
        "left_finger",
        "right_finger",
        "tool0",
        "ee_link",
        "link6",
        "base_link",
    ]
    print("\n========== BODY SEARCH ==========")
    for name in candidate_bodies:
        try:
            body_cfg = SceneEntityCfg("robot", body_names=[name])
            body_cfg.resolve(scene)
            print(f"{name}: ids={body_cfg.body_ids}, names={body_cfg.body_names}")
        except Exception:
            print(f"{name}: NOT FOUND")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
