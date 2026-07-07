#!/usr/bin/env python3
"""
修改版的 URDF 转换脚本,支持指定碰撞近似方法

使用方法:
./isaaclab.sh -p convert_urdf_with_collision.py \\
    /path/to/Door.urdf \\
    /path/to/output/Door.usd \\
    --collision-type convexDecomposition \\
    --headless
"""

import argparse
from isaaclab.app import AppLauncher

# 添加参数
parser = argparse.ArgumentParser(description="URDF 转 USD,支持指定碰撞近似方法")
parser.add_argument("input", type=str, help="输入 URDF 文件路径")
parser.add_argument("output", type=str, help="输出 USD 文件路径")
parser.add_argument(
    "--collision-type",
    type=str,
    default="convexDecomposition",
    choices=["convex_hull", "convex_decomposition"],
    help="碰撞近似方法 (默认: convexDecomposition)"
)
parser.add_argument("--merge-joints", action="store_true", default=False, help="合并固定关节")
parser.add_argument("--fix-base", action="store_true", default=False, help="固定基座")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path

def main():
    # 检查文件路径
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"无效的 URDF 文件: {urdf_path}")
    
    dest_path = os.path.abspath(args_cli.output)
    
    # 转换碰撞类型参数
    collision_map = {
        "convex_hull": "convex_hull",
        "convex_decomposition": "convex_decomposition"
    }
    collider_type = collision_map[args_cli.collision_type]
    
    # 创建转换配置
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        collider_type=collider_type,  # 关键参数!
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=100.0,
                damping=1.0,
            ),
            target_type="position",
        ),
    )
    
    print("=" * 80)
    print(f"输入 URDF: {urdf_path}")
    print(f"输出 USD: {dest_path}")
    print(f"碰撞近似: {collider_type}")
    print("=" * 80)
    
    # 执行转换
    urdf_converter = UrdfConverter(urdf_converter_cfg)
    
    print("\n✓ 转换完成!")
    print(f"生成的 USD 文件: {urdf_converter.usd_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
    simulation_app.close()
