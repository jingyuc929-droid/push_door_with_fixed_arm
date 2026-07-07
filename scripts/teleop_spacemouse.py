
"""
Script to run the environment with a SpaceMouse controller.

This script runs the environment with a SpaceMouse controller. The user can control
the robot's end-effector pose using the SpaceMouse.
"""

import argparse
import time
from isaaclab.app import AppLauncher

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="SpaceMouse Test Script")
parser.add_argument("--task", type=str, default="Template-Door-Env-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动 AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.envs import ManagerBasedRLEnv
from door_env.tasks.manager_based.door_env.door_env_env_cfg import DoorEnvEnvCfg
from isaaclab.managers import SceneEntityCfg

class CustomSe3SpaceMouse(Se3SpaceMouse):
    """自定义 SpaceMouse 类以支持更多型号并调试数据。"""
    def _find_device(self):
        """覆盖原始方法以包含 'SpaceMouse Wireless BT'。"""
        import hid
        import time
        found = False
        for _ in range(5):
            for device in hid.enumerate():
                if (
                    device["product_string"] == "SpaceMouse Compact"
                    or device["product_string"] == "SpaceMouse Wireless"
                    or device["product_string"] == "3Dconnexion Universal Receiver"
                    or device["product_string"] == "SpaceMouse Wireless BT"
                ):
                    found = True
                    vendor_id = device["vendor_id"]
                    product_id = device["product_id"]
                    self._device.close()
                    self._device.open(vendor_id, product_id)
                    self._device_name = device["product_string"]
            if not found:
                time.sleep(1.0)
            else:
                break
        if not found:
            raise OSError("No device found by SpaceMouse. Is the device connected?")

    def _run_device(self):
        """覆盖监听线程以打印调试信息。"""
        from isaaclab.devices.spacemouse.utils import convert_buffer
        while True:
            data = self._device.read(13) # 增加读取长度以防万一
            if data:
                # 解析数据包
                
                if data[0] == 1:
                    # 平移数据 (字节 1-6)
                    self._delta_pos[1] = self.pos_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_pos[0] = self.pos_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_pos[2] = self.pos_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                    
                    # 如果数据包长度为 13，说明旋转数据也在其中 (字节 7-12)
                    if len(data) >= 13:
                        self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[7], data[8])
                        self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[9], data[10])
                        self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[11], data[12]) * -1.0
                elif data[0] == 2:
                    # 传统的独立旋转数据包
                    self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                elif data[0] == 3:
                    # 按钮数据
                    if data[1] == 1:
                        self._close_gripper = not self._close_gripper
                    if data[1] == 2:
                        self.reset()

def main():
    # 配置 SpaceMouse
    sm_cfg = Se3SpaceMouseCfg(
        pos_sensitivity=0.05,
        rot_sensitivity=0.05,
    )
    sm_cfg.class_type = CustomSe3SpaceMouse
    try:
        # 使用配置中的 class_type 实例化
        device = sm_cfg.class_type(sm_cfg)
    except OSError as e:
        print(f"Error: {e}")
        return

    print(device)
    print("开始读取数据... (按 Ctrl+C 退出)")

    # Load environment config
    env_cfg = DoorEnvEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device else "cuda:0"

    # Create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Initialize variables
    count = 0 
    
    print("\nStarting simulation loop... Press Ctrl+C to stop.")
    try:
        while simulation_app.is_running():
            # 获取当前指令 [x, y, z, rx, ry, rz, gripper]
            # Advance the controller
            command = device.advance()
            
            # Convert to torch tensor if it's not already (Se3SpaceMouse returns cpu tensor)
            if isinstance(command, torch.Tensor):
                command_tensor = command.to(device=env.device)
            else:
                command_tensor = torch.tensor(command, dtype=torch.float32, device=env.device)
            
            # Split command
            arm_command = command_tensor[:6]
            gripper_command = command_tensor[6].unsqueeze(0) # Make it 1D

            # Combine into single tensor for step
            actions = torch.cat([arm_command, gripper_command], dim=0).unsqueeze(0)
            
            # Step environment
            obs, rew, terminated, truncated, info = env.step(actions)
            
            # time.sleep(0.01) # Env step controls timing
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        env.close()
        simulation_app.close()

if __name__ == "__main__":
    main()
