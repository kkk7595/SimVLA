import h5py
import numpy as np
import os
import cv2

# ===================== 【你的 HDF5 路径】 =====================
# HDF5_PATH = "/home/franka/lqz/hkx/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
HDF5_PATH = "/home/keep/Desktop/project/X-RLinf/dataset/franka_hdf5/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5"

# 图片保存根目录
SAVE_BASE_DIR = "/home/keep/Desktop/project/X-RLinf"
SAVE_IMG_DIR = os.path.join(SAVE_BASE_DIR, "hdf5_export_images")
os.makedirs(SAVE_IMG_DIR, exist_ok=True)
# =================================================================

def analyze_libero_hdf5(hdf5_path):
    """分析 LIBERO HDF5 数据集：遍历所有 demo，自动适配，不写死路径"""
    if not os.path.exists(hdf5_path):
        print(f"❌ 文件不存在: {hdf5_path}")
        return

    print("=" * 80)
    print(f"📊 开始分析 LIBERO HDF5 数据: {hdf5_path}")
    print("=" * 80)

    with h5py.File(hdf5_path, "r") as f:
        # --------------------- 1. 打印顶层属性 ---------------------
        print("\n✅ 顶层属性 (attrs)：")
        for key in f.attrs:
            print(f"  {key}: {f.attrs[key]}")

        # --------------------- 2. 打印完整结构 ---------------------
        print("\n✅ HDF5 完整结构：")
        def print_name(name):
            if isinstance(f[name], h5py.Dataset):
                print(f"  📄 {name} | 形状: {f[name].shape} | 类型: {f[name].dtype}")
            else:
                print(f"  📂 {name} (组)")
        f.visit(print_name)

        # 🔥🔥🔥 核心修复：自动获取所有 demo，遍历全部，不写死 demo_9
        if "data" not in f:
            print("❌ 没有 data 组！")
            return
        
        demo_names = list(f["data"].keys())  # 获取所有 demo_0, demo_1, demo_9...
        print(f"\n✅ 找到全部 {len(demo_names)} 个 demo: {demo_names}")

        # 🔥🔥🔥 遍历每一个 demo 处理
        for demo_name in demo_names:
            demo_path = f"data/{demo_name}"
            print("\n" + "=" * 80)
            print(f"🔍 正在处理：{demo_path}")
            print("=" * 80)

            # --------------------- 读取 obs ---------------------
            print(f"\n✅ 【{demo_name}】观测数据：")
            obs = f[f"{demo_path}/obs"]

            # 相机图像
            if "agentview_rgb" in obs:
                img_agent = obs["agentview_rgb"][:]
                print(f"  agentview_rgb: 形状={img_agent.shape}, 范围=[{img_agent.min()}, {img_agent.max()}]")

            if "eye_in_hand_rgb" in obs:
                img_eye = obs["eye_in_hand_rgb"][:]
                print(f"  eye_in_hand_rgb: 形状={img_eye.shape}, 范围=[{img_eye.min()}, {img_eye.max()}]")

            # 🔥 按 demo 分文件夹保存图片，避免覆盖
            demo_img_dir = os.path.join(SAVE_IMG_DIR, demo_name)
            os.makedirs(demo_img_dir, exist_ok=True)

            # 逐帧保存 agentview_rgb
            if "agentview_rgb" in obs:
                agentview_frames = obs["agentview_rgb"][:]
                total_frame = agentview_frames.shape[0]
                for idx in range(total_frame):
                    frame = agentview_frames[idx]
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(demo_img_dir, f"agentview_rgb_{idx:05d}.jpg")
                    cv2.imwrite(save_path, frame_bgr)
                print(f"💾 已保存 agentview 至: {demo_img_dir}")

            # 逐帧保存 eye_in_hand_rgb
            if "eye_in_hand_rgb" in obs:
                eye_in_hand_frames = obs["eye_in_hand_rgb"][:]
                total_frame = eye_in_hand_frames.shape[0]
                for idx in range(total_frame):
                    frame = eye_in_hand_frames[idx]
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(demo_img_dir, f"eye_in_hand_rgb_{idx:05d}.jpg")
                    cv2.imwrite(save_path, frame_bgr)
                print(f"💾 已保存 eye_in_hand 至: {demo_img_dir}")

            # 关节 / 末端状态
            if "joint_states" in obs:
                joint = obs["joint_states"][:]
                print(f"  joint_states (关节): 形状={joint.shape}, 范围=[{joint.min():.3f}, {joint.max():.3f}]")

            if "ee_pos" in obs:
                ee_pos = obs["ee_pos"][:]
                print(f"  ee_pos (末端位置): 形状={ee_pos.shape}")

            if "ee_ori" in obs:
                ee_ori = obs["ee_ori"][:]
                print(f"  ee_ori (末端姿态): 形状={ee_ori.shape}")

            # --------------------- 读取存在的数据（无 arm_actions） ---------------------
            print(f"\n✅ 【{demo_name}】机器人状态数据：")

            if "robot_states" in f[demo_path]:
                robot_states = f[f"{demo_path}/robot_states"][:]
                print(f"  robot_states: 形状={robot_states.shape}")

            if "states" in f[demo_path]:
                states = f[f"{demo_path}/states"][:]
                print(f"  states: 形状={states.shape}")

            if "rewards" in f[demo_path]:
                rewards = f[f"{demo_path}/rewards"][:]
                print(f"  rewards: 形状={rewards.shape}")

        print("\n🎉 所有 demo 处理完成！全部图片已按 demo 分文件夹保存！")

# 运行分析
if __name__ == "__main__":
    analyze_libero_hdf5(HDF5_PATH)