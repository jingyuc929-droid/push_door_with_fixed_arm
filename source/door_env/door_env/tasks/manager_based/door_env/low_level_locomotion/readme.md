我想把一个冻结的 low-level locomotion policy 集成到当前上层任务中。目标是：上层策略输出 base command + arm/gripper action，low-level policy 只负责根据 base command 控制腿部 12 个关节。

参考项目中的关键逻辑：

1. 核心文件
- common/pick_place/actions.py
  需要参考/迁移里面的：
  - _CheckpointLowLevelPolicy
  - _OnnxLowLevelPolicy 可选
  - HighLevelPickPlaceAction / HighLevelPickPlaceActionCfg 的 low-level 部分
  - _build_low_level_obs
  - _update_low_level_target
  - apply_actions
  - reset

- common/pick_place/observations.py
  可选迁移：
  - high_level_base_command
  - high_level_previous_action
  - low_level_last_action

2. checkpoint
low-level policy 是一个 .pt checkpoint，包含：
- checkpoint["model_state_dict"] 对应 ActorCriticEncoder
- checkpoint["vae_state_dict"] 对应 VAEBlind

如果目标项目没有 rl_algorithms，需要一起带上：
- rl_algorithms.rsl_rl.modules.ActorCriticEncoder
- rl_algorithms.rsl_rl.modules.VAEBlind
- 以及它们依赖的 rl_algorithms 模块
- tensordict.TensorDict

3. low-level policy 配置
需要复制这些配置字段：

low_level_policy_path = "/path/to/model_30000.pt"
low_level_policy_format = "checkpoint"

base_command_low = (-0.45, -0.25, -0.5, 0.38, -0.15)
base_command_high = (0.8, 0.25, 0.5, 0.47, 0.28)
command_smoothing_alpha = 0.35

low_level_obs_dim = 47
low_level_history_length = 5
low_level_action_scale = 0.25
low_level_decimation = 8
low_level_base_ang_vel_scale = 0.25
low_level_joint_vel_scale = 0.05
low_level_last_action_scale = 0.25
low_level_command_scale = (2.0, 2.0, 0.25, 2.0, 1.0)

还需要复制 low_level_actor_cfg 和 low_level_vae_cfg，必须和 checkpoint 训练时结构一致。

4. low-level 输入语义
low-level obs 是 47 维：

3  root_ang_vel_b * 0.25
3  projected_gravity_b
12 leg_joint_pos - default_leg_joint_pos
12 leg_joint_vel * 0.05
12 last_low_action * 0.25
5  base_command * (2.0, 2.0, 0.25, 2.0, 1.0)

base_command 的 5 维含义是：
[vx, vy, wz, body_height, body_pitch]

所以 high-level policy 的前 5 维动作应该映射成这个 command。low-level 不知道任务目标，只跟踪这 5 维命令。

5. low-level 输出语义
low-level 输出 12 维腿部动作：

low_action = policy(low_obs, low_obs_history)

最终腿关节目标为：

leg_target = default_leg_joint_pos + low_action * low_level_action_scale

也就是：
leg_target = default_leg_joint_pos + low_action * 0.25

6. 在目标任务中如何嵌入
建议新建自己的 ActionTerm，例如：

HighLevelDoorOpenAction
HighLevelDoorOpenActionCfg

不要直接叫 HighLevelPickPlaceAction。可以复制它的 low-level 部分，然后把上层 arm/gripper/door action 逻辑替换成当前任务需要的逻辑。

环境配置中要替换默认 action：

self.commands.base_command = None
self.actions.joint_pos = None
self.actions.high_level = HighLevelDoorOpenActionCfg(
    asset_name="robot",
    leg_joint_names=ROBOT_JOINT_NAMES,
    arm_joint_names=ROBOT_ARM_JOINT_NAMES,
    low_level_policy_path=...,
    low_level_policy_format="checkpoint",
    low_level_actor_cfg=...,
    low_level_vae_cfg=...,
    low_level_obs_dim=47,
    low_level_history_length=5,
    low_level_action_scale=0.25,
    low_level_decimation=8,
    low_level_command_scale=(2.0, 2.0, 0.25, 2.0, 1.0),
    command_smoothing_alpha=0.35,
)

7. 需要保持一致的地方
- 机器人腿部 12 个关节名和顺序必须和 checkpoint 训练时一致。
- low-level obs 维度必须是 47。
- VAE history 必须是 5 帧。
- low-level actor cfg / VAE cfg 必须和 checkpoint 匹配。
- high-level action 前 5 维必须能转成 [vx, vy, wz, body_height, body_pitch]。
- low-level 输出只控制腿，不控制机械臂/夹爪。
- 如果机器人质量、机械臂安装、默认姿态变化很大，checkpoint 可能能跑但效果不保证。

8. 最小迁移目标
把 low-level 当成冻结的腿部执行器：
- high-level 决定走向哪里、身体高度/俯仰、机械臂/夹爪动作；
- low-level 只根据 base command 和当前腿部状态输出 12 个腿关节目标。
