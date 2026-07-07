import argparse

from isaaclab.app import AppLauncher

# 启动 Isaac Sim / Isaac Lab
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg

from source.robot.arx5 import ARX5_CFG   # 这里改成你自己的实际包路径


@configclass
class DebugSceneCfg(InteractiveSceneCfg):
    robot = ARX5_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    num_envs = 1
    env_spacing = 2.0
    replicate_physics = False


def main():
    sim_cfg = SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 1.5], [0.0, 0.0, 0.5])

    # 地面
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # 光照
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    scene_cfg = DebugSceneCfg()
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot: Articulation = scene["robot"]

    print("\n========== ROBOT BASIC INFO ==========")
    print("is_fixed_base =", robot.is_fixed_base)
    print("num_joints    =", robot.num_joints)
    print("num_bodies    =", robot.num_bodies)

    print("\n========== JOINT NAMES ==========")
    for i, name in enumerate(robot.joint_names):
        print(f"{i:02d}: {name}")

    print("\n========== BODY NAMES ==========")
    for i, name in enumerate(robot.body_names):
        print(f"{i:02d}: {name}")

    # 尝试解析机械臂主关节
    arm_cfg = SceneEntityCfg("robot", joint_names=["joint[1-6]"], preserve_order=True)
    arm_cfg.resolve(scene)
    print("\n========== ARM JOINT RESOLVE ==========")
    print("joint_ids   =", arm_cfg.joint_ids)
    print("joint_names =", arm_cfg.joint_names)

    # 尝试解析 gripper joint
    grip_cfg = SceneEntityCfg("robot", joint_names=["gripper_joint"])
    grip_cfg.resolve(scene)
    print("\n========== GRIPPER JOINT RESOLVE ==========")
    print("joint_ids   =", grip_cfg.joint_ids)
    print("joint_names =", grip_cfg.joint_names)

    # 尝试找一些可能的末端 / 夹爪 body
    candidate_bodies = [
        "gripper_center",
        "left_pad",
        "right_pad",
        "left_finger",
        "right_finger",
        "tool0",
        "ee_link",
        "link6",
    ]
    print("\n========== BODY SEARCH ==========")
    for name in candidate_bodies:
        try:
            body_cfg = SceneEntityCfg("robot", body_names=[name])
            body_cfg.resolve(scene)
            print(f"{name}: ids={body_cfg.body_ids}, names={body_cfg.body_names}")
        except Exception as e:
            print(f"{name}: NOT FOUND")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim_cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()