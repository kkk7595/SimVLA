import pinocchio as pin
import numpy as np

class FrankaFK:
    def __init__(self, urdf_path: str="/home/keep/Desktop/project/X-RLinf/rlinf/models/embodiment/simvla/datasets/urdf/fr3_franka_hand.urdf"):
        """
        初始化 Franka 正运动学
        :param urdf_path: 你的 franka 机械臂 URDF 文件路径
        """
        # ✅ 修复：新版 Pinocchio 加载 URDF 正确写法
        self.model = pin.Model()
        pin.buildModelFromUrdf(urdf_path, self.model)
        self.data = self.model.createData()
        
        # Franka 官方末端 link 名称
        self.EE_FRAME_NAME = "fr3_hand_tcp"
        self.ee_id = self.model.getFrameId(self.EE_FRAME_NAME)

        print(f"✅ 加载 URDF 成功")
        print(f"✅ 末端执行器: {self.EE_FRAME_NAME}")
        print(f"✅ 关节数量: {self.model.nq}")

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        正运动学：关节角 → 末端位姿
        :param q: 9维关节角数组
        :return: pos(3), quat(xyzw), euler(rpy)
        """
        q_pin = pin.neutral(self.model)
        q_pin[:] = q

        # 计算正运动学
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacement(self.model, self.data, self.ee_id)

        # 提取位姿
        T = self.data.oMf[self.ee_id]
        pos = T.translation.copy()
        rot_mat = T.rotation.copy()
        quat = pin.Quaternion(T.rotation).coeffs()
        euler = pin.rpy.matrixToRpy(rot_mat)

        return pos, quat, euler
    


import os
import h5py
import numpy as np
from tqdm import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ===================== 【你的路径配置 100% 匹配】=====================
ROOT_DIR = "/home/keep/Desktop/project/X-RLinf/dataset/franka_hand"  # 你的数据集根目录
OUTPUT_ROOT = "/home/keep/Desktop/project/X-RLinf/dataset/franka_spatial"  # 输出 HDF5 目录
SKIP_FOLDERS = ["redcube"]  # 空文件夹，跳过
CAMERA_LIST = [
    "observation.images.fixed",
    "observation.images.handeye"
]
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# =================================================================

# 获取所有需要转换的子数据集文件夹
all_subdirs = [d for d in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, d))]
dataset_folders = [d for d in all_subdirs if d not in SKIP_FOLDERS]

os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"✅ 找到 {len(dataset_folders)} 个数据集：{dataset_folders}")

# ---------------------- 逐数据集转换 ----------------------
global_ep_idx = 0  # 全局唯一 episode 编号

h5_path = os.path.join(OUTPUT_ROOT, "pick_the_red_cube_and_drop_it_in_box.hdf5")
h5_f = h5py.File(h5_path, "w")
h5_data= h5_f.create_group("data")

fk_solver = FrankaFK()

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
        joint_states = np.stack([d[STATE_KEY].numpy() for d in data_frames], axis=0)

        # joint_states 是 (N,9)，必须逐帧计算
        pos_list = []
        euler_list = []
        
        for q_single in joint_states:
            pos, quat, euler = fk_solver.fk(q_single[:9]) 
            pos_list.append(pos)
            euler_list.append(euler)
        
        ee_pos = np.array(pos_list)
        ee_ori = np.array(euler_list)
        # ====================================================================

        arm_joint = joint_states[:, :7].copy() 
        hand_joint = joint_states[:, 7:].copy() 

        # 动作指令
        actions = np.stack([d[ACTION_KEY].numpy() for d in data_frames], axis=0)

        arm_action_pos_list = []
        arm_action_euler_list = []
    
        for q_single in actions:
            pos, quat, euler = fk_solver.fk(q_single[:9]) 
            arm_action_pos_list.append(pos)
            arm_action_euler_list.append(euler)
        
        arm_action_pos = np.array(arm_action_pos_list)
        arm_action_euler = np.array(arm_action_euler_list)

        arm_actions = np.concatenate([arm_action_pos,arm_action_euler],axis=-1)
        hand_actions = actions[:, 7:].copy() 

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

        left = demo.create_group(f"left")
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