# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import torch
from isaaclab.devices.spacemouse import Se3SpaceMouse, Se3SpaceMouseCfg

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
        """覆盖设备监听线程以支持 SpaceMouse Wireless BT。"""
        # 导入需要的函数
        def convert_buffer(d1, d2):
            """Convert two bytes to a signed 16-bit integer."""
            return int.from_bytes([d1, d2], byteorder='little', signed=True)
        
        while True:
            # 读取设备数据
            if self._device_name == "3Dconnexion Universal Receiver":
                data = self._device.read(7 + 6)
            elif self._device_name == "SpaceMouse Wireless BT":
                # SpaceMouse Wireless BT 可能需要不同的读取大小
                data = self._device.read(13)
            else:
                data = self._device.read(7)
            
            if data is not None:
                # 打印完整数据包用于调试
                if data[0] in [1, 2]:
                    if len(data) >= 13:
                        print(f"[SpaceMouse Debug] data[0]={data[0]}, full_data={list(data[:13])}")
                    else:
                        print(f"[SpaceMouse Debug] data[0]={data[0]}, data={list(data[:7])}")
                
                # 处理位置数据
                if data[0] == 1:
                    self._delta_pos[1] = self.pos_sensitivity * convert_buffer(data[1], data[2]) * -1.0  # 反转左右方向
                    self._delta_pos[0] = self.pos_sensitivity * convert_buffer(data[3], data[4]) * -1.0  # 反转前后方向
                    self._delta_pos[2] = self.pos_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                    
                    # 尝试从同一数据包读取旋转数据(如果有的话)
                    if len(data) >= 13:
                        self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[7], data[8]) * -1.0  # 反转俯仰角
                        self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[9], data[10]) * -1.0  # 反转横滚角
                        self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[11], data[12]) * -1.0
                
                # 处理旋转数据 - 移除 self._read_rotation 检查
                elif data[0] == 2:
                    self._delta_rot[1] = self.rot_sensitivity * convert_buffer(data[1], data[2])
                    self._delta_rot[0] = self.rot_sensitivity * convert_buffer(data[3], data[4])
                    self._delta_rot[2] = self.rot_sensitivity * convert_buffer(data[5], data[6]) * -1.0
                
                # 处理按钮
                elif data[0] == 3:
                    if data[1] == 1:  # 左按钮
                        self._close_gripper = not self._close_gripper
                        if "L" in self._additional_callbacks:
                            self._additional_callbacks["L"]()
                    if data[1] == 2:  # 右按钮
                        self.reset()
                        if "R" in self._additional_callbacks:
                            self._additional_callbacks["R"]()

class TeleopWrapper(gym.Wrapper):
    """Wrapper to override agent actions with SpaceMouse input."""

    def __init__(self, env: gym.Env, device_cfg: Se3SpaceMouseCfg = None):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap.
            device_cfg: The configuration for the SpaceMouse.
        """
        super().__init__(env)
        
        # Default config if none provided
        if device_cfg is None:
            device_cfg = Se3SpaceMouseCfg(
                pos_sensitivity=0.01,  # 进一步降低位置灵敏度
                rot_sensitivity=0.02,  # 进一步降低旋转灵敏度
            )
        
        # Force correct class type
        device_cfg.class_type = CustomSe3SpaceMouse
        
        # Initialize device
        try:
            self.device = device_cfg.class_type(device_cfg)
            print(f"[INFO] TeleopWrapper: SpaceMouse initialized: {self.device}")
        except OSError as e:
            print(f"[ERROR] TeleopWrapper: Failed to initialize SpaceMouse: {e}")
            raise e

    def step(self, action):
        """Override the action with SpaceMouse input."""
        # Get command from SpaceMouse
        command = self.device.advance()
        
        # Convert to torch tensor if needed
        if isinstance(command, torch.Tensor):
            command_tensor = command.to(device=self.env.unwrapped.device)
        else:
            command_tensor = torch.tensor(command, dtype=torch.float32, device=self.env.unwrapped.device)
            
        # Debug: print command values
        if torch.any(torch.abs(command_tensor) > 0.001):  # Only print if there's significant input
            print(f"[SpaceMouse] Pos: [{command_tensor[0]:.3f}, {command_tensor[1]:.3f}, {command_tensor[2]:.3f}] "
                  f"Rot: [{command_tensor[3]:.3f}, {command_tensor[4]:.3f}, {command_tensor[5]:.3f}] "
                  f"Gripper: {command_tensor[6]:.3f}")
        
        # Split command
        # Expected action shape: (num_envs, 7)
        # command is (7,) -> [x, y, z, rx, ry, rz, gripper]
        
        arm_command = command_tensor[:6]
        gripper_command = command_tensor[6].unsqueeze(0)
        
        # Create action for all environments (broadcasting)
        # Assuming single agent control broadcasted to all envs if num_envs > 1
        num_envs = self.env.unwrapped.num_envs
        new_action = torch.cat([arm_command, gripper_command], dim=0).unsqueeze(0).repeat(num_envs, 1)
        
        return self.env.step(new_action)

    def close(self):
        """Close the device."""
        if hasattr(self, "device"):
            del self.device
        return self.env.close()
