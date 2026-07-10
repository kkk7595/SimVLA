import numpy as np
import os
import h5py
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ===================== 【你的路径配置 100% 匹配】=====================
ROOT_DIR = "/home/franka/hkx/Data/franka_hand"  # 你的数据集根目录
OUTPUT_ROOT = "/home/franka/hkx/Data/franka_spatial"  # 输出 HDF5 目录
SKIP_FOLDERS = ["redcube"]  # 空文件夹，跳过
CAMERA_LIST = [
    "observation.images.fixed",
    "observation.images.handeye"
]
STATE_KEY = "observation.state"
ACTION_KEY = "action"
H5_FILENAME = "pick_the_yellow_cube_and_drop_it_in_the_box_demo"

# =================================================================

# 获取所有需要转换的子数据集文件夹
all_subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
dataset_folders = [d for d in all_subdirs if H5_FILENAME in d]

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"✅ 找到 {len(dataset_folders)} 个数据集：{dataset_folders}")

# ---------------------- 逐数据集转换 ----------------------
global_ep_idx = 0  # 全局唯一 episode 编号

h5_path = os.path.join(OUTPUT_ROOT, f"{H5_FILENAME}.hdf5")
h5_f = h5py.File(h5_path, "w")
h5_data= h5_f.create_group("data")


for dataset_name in dataset_folders:
    dataset_path = os.path.join(ROOT_DIR, dataset_name)
    print(f"\n===== 正在处理：{dataset_name} =====")

    try:
        # 加载 LeRobot 本地数据集（自动读取 parquet + videos）
        dataset = LeRobotDataset(
            repo_id=f"local/{dataset_name}",
            root=dataset_path,
            delta_timestamps=None
        )
    except Exception as e:
        print(f"❌ 加载失败，跳过：{dataset_name} | 错误：{e}")
        continue

    # 获取所有 episode 的起止帧
    episodes = dataset.meta.episodes
    num_eps = len(episodes)
    print(f"✅ 加载成功，{dataset_name} 包含 {num_eps} 个片段")

    # 逐 episode 转换
    for ep in tqdm(episodes, total=num_eps, desc=dataset_name):
        from_idx = int(ep["dataset_from_index"])
        to_idx = int(ep["dataset_to_index"])
        num_frames = to_idx - from_idx

        # 读取本片段所有帧
        data_frames = [dataset[i] for i in range(from_idx, to_idx)]

        # ===================== 提取数据 =====================
        # 关节状态 qpos
        observation_states = np.stack([d[STATE_KEY].numpy() for d in data_frames], axis=0)

        arm_joint = observation_states[:, :7].copy() 
        hand_joint = observation_states[:, 7:13].copy() 
        ee_pos = observation_states[:, 13:16].copy()
        ee_ori = observation_states[:, 16:19].copy()

        # 动作指令
        actions = np.stack([d[ACTION_KEY].numpy() for d in data_frames], axis=0)

        hand_actions = actions[:, 7:13].copy() 
        arm_actions = actions[:, 13:19].copy()

        # done 信号）
        dones = np.zeros(len(data_frames), dtype=np.uint8)
        dones[-1] = 1

        # #
        rewards = dones.copy() 

        # 双相机图像 (T, H, W, 3) uint8
        imgs_fixed = []
        imgs_handeye = []
        for d in data_frames:
            # fixed 相机
            img = d[CAMERA_LIST[0]].permute(1, 2, 0).numpy()  # C H W → H W C
            img = (img * 255).astype(np.uint8)
            imgs_fixed.append(img)

            # handeye 相机
            img = d[CAMERA_LIST[1]].permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            imgs_handeye.append(img)

        imgs_fixed = np.stack(imgs_fixed, axis=0)
        imgs_handeye = np.stack(imgs_handeye, axis=0)

        # ===================== 写入 LIBRO 标准 HDF5 =====================
        demo= h5_data.create_group(f"demo_{global_ep_idx}")

        left = demo.create_group(f"right")
        left.create_dataset("arm_actions", data=arm_actions, compression="gzip")
        left.create_dataset("hand_actions", data=hand_actions, compression="gzip")
        left.create_dataset("dones", data=dones, compression="gzip")
        left.create_dataset("rewards", data=rewards, compression="gzip")
        
        obs= left.create_group("obs")
        obs.create_dataset("ee_ori", data=ee_ori, compression="gzip")
        obs.create_dataset("ee_pos", data=ee_pos, compression="gzip")
        obs.create_dataset("arm_joint", data=arm_joint, compression="gzip")
        obs.create_dataset("hand_joint", data=hand_joint, compression="gzip")
        obs.create_dataset("eye_in_hand_rgb", data=imgs_handeye, compression="gzip")
        obs.create_dataset("agentview_rgb", data=imgs_fixed, compression="gzip")

        global_ep_idx += 1

    print(f"🎉 {dataset_name} 转换完成！现计 {global_ep_idx} 个片段")
h5_f.close()
print(f"\n🎉 全部转换完成！总计 {global_ep_idx} 个片段，输出目录：{OUTPUT_ROOT}")