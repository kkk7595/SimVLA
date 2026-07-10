from __future__ import annotations

import io
import random
import glob
import os
import re
from typing import Optional, Tuple, Iterable, Sequence, Any, Dict, List

import numpy as np
import h5py
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R  # 姿态转换工具

from .base import DomainHandler


def _quat2axisangle_single(quat: np.ndarray) -> np.ndarray:
    """
    将单个四元数 [x,y,z,w] 转换为轴角表示
    遵循robosuite实现，确保与推理阶段一致
    
    Args:
        quat: 四元数数组 [x,y,z,w]
    
    Returns:
        axis_angle: 轴角表示数组 [x,y,z]
    """
    import math
    quat = quat.copy()
    # 限制四元数w分量范围（避免数值不稳定）
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    
    den = np.sqrt(1.0 - quat[3] * quat[3])
    # 避免除以0（当四元数接近单位四元数时）
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    
    # 四元数转轴角公式：axis = (x,y,z) * 2*acos(w) / sqrt(1-w²)
    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def euler_to_axisangle(euler: np.ndarray) -> np.ndarray:
    """
    将欧拉角（XYZ顺序）转换为轴角表示
    转换流程：欧拉角 → 四元数 → 轴角
    
    Args:
        euler: [T, 3] 欧拉角数组（roll, pitch, yaw）
    
    Returns:
        axis_angle: [T, 3] 轴角表示数组
    """
    # 欧拉角转旋转对象
    rot = R.from_euler('xyz', euler)
    # 旋转对象转四元数 [T, 4]（格式：x,y,z,w）
    quats = rot.as_quat()
    
    # 处理单帧情况
    if quats.ndim == 1:
        return _quat2axisangle_single(quats)
    
    # 批量转换
    axis_angles = np.zeros((len(quats), 3), dtype=np.float32)
    for i in range(len(quats)):
        axis_angles[i] = _quat2axisangle_single(quats[i])
    return axis_angles


class LiberoHDF5Handler(DomainHandler):
    """
    LIBERO原始HDF5数据集处理器
    直接读取LIBERO官方HDF5格式，支持libero_10、libero_90、libero_goal、libero_object、libero_spatial等子集
    
    核心功能：
    - 解析HDF5中的动作、图像、 proprioception（本体感受）数据
    - 将欧拉角转换为轴角表示
    - 生成模型训练所需的样本格式
    """
    dataset_name = "libero_hdf5"
    
    # 数据帧率和预测窗口时长
    FREQ = 10.0  # 10Hz
    QDUR = 1.0   # 1秒
    
    def __init__(self, meta: dict, num_views: int = 3) -> None:
        """
        初始化LIBERO处理器
        Args:
            meta: 数据集元信息（包含数据目录、datalist等）
            num_views: 多视图图像数量（默认3，实际使用agentview和wrist两个视图）
        """
        super().__init__(meta, num_views)
        self.data_dir = meta.get("data_dir", "")
        self.h5_files: List[str] = []  # HDF5文件路径列表
        self.task_names: List[str] = []  # 任务指令列表
        
        # 从datalist中加载HDF5文件和任务名称
        if "datalist" in meta:
            for item in meta["datalist"]:
                if isinstance(item, dict):
                    self.h5_files.append(item["path"])
                    self.task_names.append(item.get("task", ""))
                else:
                    self.h5_files.append(item)
                    self.task_names.append(self._parse_task_from_filename(item))
        
    def _parse_task_from_filename(self, filepath: str) -> str:
        """
        从文件名解析任务描述
        示例：KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 → "turn on the stove and put the moka pot on it"
        """
        base = os.path.basename(filepath)
        # 移除后缀 "_demo.hdf5"
        task = re.sub(r"_demo\.hdf5$", "", base)
        # 移除场景前缀（如"SCENE3_"）
        m = re.search(r"SCENE\d+_", task)
        if m:
            task = task[m.end():]
        # 下划线转空格
        task = task.replace("_", " ")
        return task
    
    def _open_h5(self, path: str) -> h5py.File:
        """打开HDF5文件（本地）"""
        return h5py.File(path, "r")

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "libero_joint",
        lang_aug_map: dict | None = None,
        **kwargs
    ) -> Iterable[dict]:
        """
        遍历单个episode（轨迹）的所有样本
        
        Args:
            traj_idx: 轨迹索引
            num_actions: 动作序列长度（默认10）
            training: 是否为训练模式（影响数据增强和采样）
            image_aug: 图像增强变换
            action_mode: 动作模式（仅支持libero_joint）
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典，包含语言指令、图像、本体感受、动作轨迹等
        """
        h5_path = self.h5_files[traj_idx]
        task_instruction = self.task_names[traj_idx]
        
        # 安全打开HDF5文件
        with self._open_h5(h5_path) as f:
            if "data" not in f:
                return
            data_grp = f["data"]
            
            # 获取所有demo键，训练时随机打乱
            demo_keys = list(data_grp.keys())
            if training:
                random.shuffle(demo_keys)
            
            # 遍历每个demo
            for demo_key in demo_keys:
                demo = data_grp[demo_key]
                
                # 检查必要键是否存在
                required_keys = ["actions", "obs/agentview_rgb", "obs/eye_in_hand_rgb"]
                if not all(k in demo or f"obs/{k.split('/')[-1]}" in demo.get("obs", {}) 
                          for k in required_keys if "/" not in k):
                    continue
                
                try:
                    # 处理单个demo并生成样本
                    yield from self._iter_demo(
                        demo,
                        task_instruction,
                        num_actions=num_actions,
                        training=training,
                        image_aug=image_aug,
                        action_mode=action_mode,
                        lang_aug_map=lang_aug_map,
                    )
                except Exception as e:
                    print(f"处理 {h5_path}/{demo_key} 时出错: {e}")
                    continue
    
    def _iter_demo(
        self,
        demo: h5py.Group,
        task_instruction: str,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode: str,
        lang_aug_map: dict | None,
    ) -> Iterable[dict]:
        """
        处理单个demo，生成样本迭代器
        
        Args:
            demo: HDF5中的demo组
            task_instruction: 任务指令
            num_actions: 动作序列长度
            training: 是否训练模式
            image_aug: 图像增强
            action_mode: 动作模式
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典
        """
        # 加载动作数据 [T, 7]（delta_xyz(3) + delta_euler(3) + gripper(1)）
        arm_actions = np.array(demo["left/arm_actions"])
        hand_actions = np.array(demo["left/hand_actions"])

        # 加载图像数据
        agentview_rgb = np.array(demo["left/obs/agentview_rgb"])  # [T, H, W, 3] 第三人称视图
        wrist_rgb = np.array(demo["left/obs/eye_in_hand_rgb"])     # [T, H, W, 3] 腕部视图
        
        # 加载本体感受数据
        ee_pos = np.array(demo["left/obs/ee_pos"])  # [T, 3] 末端执行器位置
        ee_ori_euler = np.array(demo["left/obs/ee_ori"])  # [T, 3] 末端执行器欧拉角
        arm_joint = np.array(demo["left/obs/arm_joint"])  # [T, 7] 机械臂关节角
        hand_joint = np.array(demo["left/obs/arm_joint"])  # [T, 6] 灵巧手关节角
        
        # 将欧拉角转换为轴角表示（模型输入要求）
        ee_ori_axisangle = euler_to_axisangle(ee_ori_euler)  # [T, 3]
        
        # 确定有效序列长度（取动作、图像的最小长度）
        T = min(len(actions), len(agentview_rgb), len(wrist_rgb))
        
        # 构建本体感受特征：[ee_pos(3) + axis_angle(3) + gripper(2)] → 8维
        proprio = np.concatenate([
            ee_pos[:T],
            ee_ori_axisangle[:T],
            hand_joint[:T]
        ], axis=-1).astype(np.float32)
        
        # 动作数据截取有效长度
        actions = np.concatenate([
            arm_actions[:T],
            hand_actions[:T]
        ], axis=-1).astype(np.float32)
        
        # 生成候选采样索引（确保动作序列不越界）
        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)
        
        # 图像掩码：标记有效视图（前两个为agentview和wrist）
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        
        # 遍历索引生成样本
        for idx in indices:
            # 获取动作序列片段（包含当前状态+num_actions个未来动作）
            action_chunk = self._get_action_chunk(actions, idx, num_actions)
            
            # 语言增强（训练模式且有增强映射表）
            instruction = task_instruction
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])
            
            # 处理图像：
            imgs = []
            
            # 第三人称视图：旋转180度保持一致性
            img_data = agentview_rgb[idx][::-1, ::-1].copy()
            img = Image.fromarray(img_data)
            if image_aug:
                img = image_aug(img)
            imgs.append(img)
            
            # 腕部视图：同样旋转180度
            wrist_data = wrist_rgb[idx][::-1, ::-1].copy()
            wrist_img = Image.fromarray(wrist_data)
            if image_aug:
                wrist_img = image_aug(wrist_img)
            imgs.append(wrist_img)
            
            # 填充空视图（达到num_views数量）
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            
            # 堆叠为多视图图像张量 [num_views, C, H, W]
            image_input = torch.stack(imgs, dim=0)
            
            # 生成样本字典
            yield {
                "language_instruction": instruction,  # 语言指令
                "image_input": image_input,            # 多视图图像输入
                "image_mask": image_mask,              # 图像掩码
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),  # 本体感受
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),  # 动作轨迹
            }

    def _get_action_chunk(
        self,
        actions: np.ndarray,
        start_idx: int,
        num_actions: int
    ) -> np.ndarray:
        """
        获取动作序列片段，超出范围时用最后一帧填充
        
        Returns:
            [num_actions+1, action_dim] - 包含当前状态 + num_actions个未来动作
        """
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)
        
        for i in range(num_actions + 1):
            # 限制索引不超过序列长度
            t = min(start_idx + i, T - 1)
            chunk[i] = actions[t]
        
        return chunk


class FrankaHDF5Handler(DomainHandler):
    """
    LIBERO原始HDF5数据集处理器
    直接读取LIBERO官方HDF5格式，支持libero_10、libero_90、libero_goal、libero_object、libero_spatial等子集
    
    核心功能：
    - 解析HDF5中的动作、图像、 proprioception（本体感受）数据
    - 将欧拉角转换为轴角表示
    - 生成模型训练所需的样本格式
    """
    dataset_name = "franka_hdf5"
    
    # 数据帧率和预测窗口时长
    FREQ = 30.0  # 10Hz   ## 有问题
    QDUR = 1.0   # 1秒    ## 有问题
    
    def __init__(self, meta: dict, num_views: int = 3) -> None:
        """
        初始化LIBERO处理器
        Args:
            meta: 数据集元信息（包含数据目录、datalist等）
            num_views: 多视图图像数量（默认3，实际使用agentview和wrist两个视图）
        """
        super().__init__(meta, num_views)
        self.data_dir = meta.get("data_dir", "")
        self.h5_files: List[str] = []  # HDF5文件路径列表
        self.task_names: List[str] = []  # 任务指令列表
        
        # 从datalist中加载HDF5文件和任务名称
        if "datalist" in meta:
            for item in meta["datalist"]:
                if isinstance(item, dict):
                    self.h5_files.append(item["path"])
                    self.task_names.append(item.get("task", ""))
                else:
                    self.h5_files.append(item)
                    self.task_names.append(self._parse_task_from_filename(item))
        
    def _parse_task_from_filename(self, filepath: str) -> str:
        """
        从文件名解析任务描述
        示例：KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 → "turn on the stove and put the moka pot on it"
        """
        base = os.path.basename(filepath)
        # 移除后缀 "_demo.hdf5"
        task = re.sub(r"_demo\.hdf5$", "", base)
        # 移除场景前缀（如"SCENE3_"）
        m = re.search(r"SCENE\d+_", task)
        if m:
            task = task[m.end():]
        # 下划线转空格
        task = task.replace("_", " ")
        return task
    
    def _open_h5(self, path: str) -> h5py.File:
        """打开HDF5文件（本地）"""
        return h5py.File(path, "r")
    
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "libero_joint",
        lang_aug_map: dict | None = None,
        **kwargs
    ) -> Iterable[dict]:
        """
        遍历单个episode（轨迹）的所有样本
        
        Args:
            traj_idx: 轨迹索引
            num_actions: 动作序列长度（默认10）
            training: 是否为训练模式（影响数据增强和采样）
            image_aug: 图像增强变换
            action_mode: 动作模式（仅支持libero_joint）
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典，包含语言指令、图像、本体感受、动作轨迹等
        """
        h5_path = self.h5_files[traj_idx]
        task_instruction = self.task_names[traj_idx]

        # print(f"num_actions：{num_actions}")
        # print(f"action_mode：{action_mode}")
        # print(f"h5_path：{h5_path}")
        
        # 安全打开HDF5文件
        with self._open_h5(h5_path) as f:
            if "data" not in f:
                return
            data_grp = f["data"]
            
            # 获取所有demo键，训练时随机打乱
            demo_keys = list(data_grp.keys())
            if training:
                random.shuffle(demo_keys)
            
            # 遍历每个demo
            for demo_key in demo_keys:
                demo = data_grp[demo_key]
                # print(f"demo_key：{demo_key}")
                
                # 检查必要键是否存在
                # required_keys = ["left/arm_actions", "left/hand_actions", "left/obs/agentview_rgb", "left/obs/eye_in_hand_rgb"]
                # if not all(k in demo or f"left/obs/{k.split('/')[-1]}" in demo.get("left/obs", {}) 
                #           for k in required_keys if "/" not in k):
                #     continue
                
                try:
                    # 处理单个demo并生成样本
                    yield from self._iter_demo(
                        demo,
                        task_instruction,
                        num_actions=num_actions,
                        training=training,
                        image_aug=image_aug,
                        action_mode=action_mode,
                        lang_aug_map=lang_aug_map,
                    )
                except Exception as e:
                    print(f"处理 {h5_path}/{demo_key} 时出错: {e}")
                    continue
    
    def _iter_demo(
        self,
        demo: h5py.Group,
        task_instruction: str,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode: str,
        lang_aug_map: dict | None,
    ) -> Iterable[dict]:
        """
        处理单个demo，生成样本迭代器
        
        Args:
            demo: HDF5中的demo组
            task_instruction: 任务指令
            num_actions: 动作序列长度
            training: 是否训练模式
            image_aug: 图像增强
            action_mode: 动作模式
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典
        """
        # 加载动作数据 [T, 7]（delta_xyz(3) + delta_euler(3) + gripper(1)）
        arm_actions = np.array(demo["right/arm_actions"])
        hand_actions = np.array(demo["right/hand_actions"])

        # print(f"arm_actions：{arm_actions}")

        # 加载图像数据
        agentview_rgb = np.array(demo["right/obs/agentview_rgb"])  # [T, H, W, 3] 第三人称视图
        wrist_rgb = np.array(demo["right/obs/eye_in_hand_rgb"])     # [T, H, W, 3] 腕部视图
        
        # 加载本体感受数据
        ee_pos = np.array(demo["right/obs/ee_pos"])  # [T, 3] 末端执行器位置
        ee_ori_euler = np.array(demo["right/obs/ee_ori"])  # [T, 3] 末端执行器欧拉角
        arm_joint = np.array(demo["right/obs/arm_joint"])  # [T, 7] 机械臂关节角
        hand_joint = np.array(demo["right/obs/hand_joint"])  # [T, 6] 灵巧手关节角
        
        # 将欧拉角转换为轴角表示（模型输入要求）
        ee_ori_axisangle = euler_to_axisangle(ee_ori_euler)  # [T, 3]
        
        # 确定有效序列长度（取动作、图像的最小长度）
        T = min(len(arm_actions), len(hand_actions), len(agentview_rgb), len(wrist_rgb))
        
        # 构建本体感受特征：[ee_pos(3) + axis_angle(3) + gripper(2)] → 8维
        proprio = np.concatenate([
            ee_pos[:T],
            ee_ori_axisangle[:T],
            hand_joint[:T]
        ], axis=-1).astype(np.float32)
        
        # 动作数据截取有效长度
        actions = np.concatenate([
            arm_actions[:T],
            hand_actions[:T]
        ], axis=-1).astype(np.float32)
        
        # 生成候选采样索引（确保动作序列不越界）
        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)
        
        # 图像掩码：标记有效视图（前两个为agentview和wrist）
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        
        # 遍历索引生成样本
        for idx in indices:
            # 获取动作序列片段（包含当前状态+num_actions个未来动作）
            action_chunk = self._get_action_chunk(actions, idx, num_actions)
            
            # 语言增强（训练模式且有增强映射表）
            instruction = task_instruction
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])
            
            # 处理图像：
            imgs = []
            
            # 第三人称视图：旋转180度保持一致性
            img_data = agentview_rgb[idx][::-1, ::-1].copy()
            img = Image.fromarray(img_data)
            if image_aug:
                img = image_aug(img)
            imgs.append(img)
            
            # 腕部视图：同样旋转180度
            wrist_data = wrist_rgb[idx][::-1, ::-1].copy()
            wrist_img = Image.fromarray(wrist_data)
            if image_aug:
                wrist_img = image_aug(wrist_img)
            imgs.append(wrist_img)
            
            # 填充空视图（达到num_views数量）
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            
            # 堆叠为多视图图像张量 [num_views, C, H, W]
            image_input = torch.stack(imgs, dim=0)
            
            # 生成样本字典
            yield {
                "language_instruction": instruction,  # 语言指令
                "image_input": image_input,            # 多视图图像输入
                "image_mask": image_mask,              # 图像掩码
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),  # 本体感受
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),  # 动作轨迹
            }


    def _get_action_chunk(
        self,
        actions: np.ndarray,
        start_idx: int,
        num_actions: int
    ) -> np.ndarray:
        """
        获取动作序列片段，超出范围时用最后一帧填充
        
        Returns:
            [num_actions+1, action_dim] - 包含当前状态 + num_actions个未来动作
        """
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)
        
        for i in range(num_actions + 1):
            # 限制索引不超过序列长度
            t = min(start_idx + i, T - 1)
            chunk[i] = actions[t]
        
        return chunk

class FrankaPARQUETHandler(DomainHandler):
    """
    原始PARQUET数据集处理器
    直接读取PARQUET格式，支持franka_spatial等子集
    
    核心功能：
    - 解析PARQUET中的动作、图像、 proprioception（本体感受）数据
    - 将欧拉角转换为轴角表示
    - 生成模型训练所需的样本格式
    """
    dataset_name = "franka_parquet"
    
    # 数据帧率和预测窗口时长
    FREQ = 30.0  # 10Hz   ## 有问题
    QDUR = 1.0   # 1秒    ## 有问题
    
    def __init__(self, meta: dict, num_views: int = 3) -> None:
        """
        初始化LIBERO处理器
        Args:
            meta: 数据集元信息（包含数据目录、datalist等）
            num_views: 多视图图像数量（默认3，实际使用agentview和wrist两个视图）
        """
        super().__init__(meta, num_views)
        self.data_dir = meta.get("data_dir", "")
        self.h5_files: List[str] = []  # HDF5文件路径列表
        self.task_names: List[str] = []  # 任务指令列表
        
        # 从datalist中加载HDF5文件和任务名称
        if "datalist" in meta:
            for item in meta["datalist"]:
                if isinstance(item, dict):
                    self.h5_files.append(item["path"])
                    self.task_names.append(item.get("task", ""))
                else:
                    self.h5_files.append(item)
                    self.task_names.append(self._parse_task_from_filename(item))
        
    def _parse_task_from_filename(self, filepath: str) -> str:
        """
        从文件名解析任务描述
        示例：KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5 → "turn on the stove and put the moka pot on it"
        """
        base = os.path.basename(filepath)
        # 移除后缀 "_demo.hdf5"
        task = re.sub(r"_demo\.hdf5$", "", base)
        # 移除场景前缀（如"SCENE3_"）
        m = re.search(r"SCENE\d+_", task)
        if m:
            task = task[m.end():]
        # 下划线转空格
        task = task.replace("_", " ")
        return task
    
    def _open_h5(self, path: str) -> h5py.File:
        """打开HDF5文件（本地）"""
        return h5py.File(path, "r")
    
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int = 10,
        training: bool = True,
        image_aug=None,
        action_mode: str = "libero_joint",
        lang_aug_map: dict | None = None,
        **kwargs
    ) -> Iterable[dict]:
        """
        遍历单个episode（轨迹）的所有样本
        
        Args:
            traj_idx: 轨迹索引
            num_actions: 动作序列长度（默认10）
            training: 是否为训练模式（影响数据增强和采样）
            image_aug: 图像增强变换
            action_mode: 动作模式（仅支持libero_joint）
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典，包含语言指令、图像、本体感受、动作轨迹等
        """
        h5_path = self.h5_files[traj_idx]
        task_instruction = self.task_names[traj_idx]
        
        # 安全打开HDF5文件
        with self._open_h5(h5_path) as f:
            if "data" not in f:
                return
            data_grp = f["data"]
            
            # 获取所有demo键，训练时随机打乱
            demo_keys = list(data_grp.keys())
            if training:
                random.shuffle(demo_keys)
            
            # 遍历每个demo
            for demo_key in demo_keys:
                demo = data_grp[demo_key]
                
                # 检查必要键是否存在
                required_keys = ["actions", "obs/agentview_rgb", "obs/eye_in_hand_rgb"]
                if not all(k in demo or f"obs/{k.split('/')[-1]}" in demo.get("obs", {}) 
                          for k in required_keys if "/" not in k):
                    continue
                
                try:
                    # 处理单个demo并生成样本
                    yield from self._iter_demo(
                        demo,
                        task_instruction,
                        num_actions=num_actions,
                        training=training,
                        image_aug=image_aug,
                        action_mode=action_mode,
                        lang_aug_map=lang_aug_map,
                    )
                except Exception as e:
                    print(f"处理 {h5_path}/{demo_key} 时出错: {e}")
                    continue
    
    def _iter_demo(
        self,
        demo: h5py.Group,
        task_instruction: str,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode: str,
        lang_aug_map: dict | None,
    ) -> Iterable[dict]:
        """
        处理单个demo，生成样本迭代器
        
        Args:
            demo: HDF5中的demo组
            task_instruction: 任务指令
            num_actions: 动作序列长度
            training: 是否训练模式
            image_aug: 图像增强
            action_mode: 动作模式
            lang_aug_map: 语言增强映射表
        
        Yields:
            dict: 样本字典
        """
        # 加载动作数据 [T, 7]（delta_xyz(3) + delta_euler(3) + gripper(1)）
        arm_actions = np.array(demo["left/arm_actions"])
        hand_actions = np.array(demo["left/hand_actions"])

        # 加载图像数据
        agentview_rgb = np.array(demo["left/obs/agentview_rgb"])  # [T, H, W, 3] 第三人称视图
        wrist_rgb = np.array(demo["left/obs/eye_in_hand_rgb"])     # [T, H, W, 3] 腕部视图
        
        # 加载本体感受数据
        ee_pos = np.array(demo["left/obs/ee_pos"])  # [T, 3] 末端执行器位置
        ee_ori_euler = np.array(demo["left/obs/ee_ori"])  # [T, 3] 末端执行器欧拉角
        arm_joint = np.array(demo["left/obs/arm_joint"])  # [T, 7] 机械臂关节角
        hand_joint = np.array(demo["left/obs/arm_joint"])  # [T, 6] 灵巧手关节角
        
        # 将欧拉角转换为轴角表示（模型输入要求）
        ee_ori_axisangle = euler_to_axisangle(ee_ori_euler)  # [T, 3]
        
        # 确定有效序列长度（取动作、图像的最小长度）
        T = min(len(actions), len(agentview_rgb), len(wrist_rgb))
        
        # 构建本体感受特征：[ee_pos(3) + axis_angle(3) + gripper(2)] → 8维
        proprio = np.concatenate([
            ee_pos[:T],
            ee_ori_axisangle[:T],
            hand_joint[:T]
        ], axis=-1).astype(np.float32)
        
        # 动作数据截取有效长度
        actions = np.concatenate([
            arm_actions[:T],
            hand_actions[:T]
        ], axis=-1).astype(np.float32)
        
        # 生成候选采样索引（确保动作序列不越界）
        indices = list(range(max(0, T - num_actions)))
        if training:
            random.shuffle(indices)
        
        # 图像掩码：标记有效视图（前两个为agentview和wrist）
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:2] = True
        
        # 遍历索引生成样本
        for idx in indices:
            # 获取动作序列片段（包含当前状态+num_actions个未来动作）
            action_chunk = self._get_action_chunk(actions, idx, num_actions)
            
            # 语言增强（训练模式且有增强映射表）
            instruction = task_instruction
            if training and lang_aug_map and instruction in lang_aug_map:
                instruction = random.choice(lang_aug_map[instruction])
            
            # 处理图像：
            imgs = []
            
            # 第三人称视图：旋转180度保持一致性
            img_data = agentview_rgb[idx][::-1, ::-1].copy()
            img = Image.fromarray(img_data)
            if image_aug:
                img = image_aug(img)
            imgs.append(img)
            
            # 腕部视图：同样旋转180度
            wrist_data = wrist_rgb[idx][::-1, ::-1].copy()
            wrist_img = Image.fromarray(wrist_data)
            if image_aug:
                wrist_img = image_aug(wrist_img)
            imgs.append(wrist_img)
            
            # 填充空视图（达到num_views数量）
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            
            # 堆叠为多视图图像张量 [num_views, C, H, W]
            image_input = torch.stack(imgs, dim=0)
            
            # 生成样本字典
            yield {
                "language_instruction": instruction,  # 语言指令
                "image_input": image_input,            # 多视图图像输入
                "image_mask": image_mask,              # 图像掩码
                "proprio": torch.tensor(proprio[idx], dtype=torch.float32),  # 本体感受
                "abs_trajectory": torch.tensor(action_chunk, dtype=torch.float32),  # 动作轨迹
            }

    def _get_action_chunk(
        self,
        actions: np.ndarray,
        start_idx: int,
        num_actions: int
    ) -> np.ndarray:
        """
        获取动作序列片段，超出范围时用最后一帧填充
        
        Returns:
            [num_actions+1, action_dim] - 包含当前状态 + num_actions个未来动作
        """
        T, action_dim = actions.shape
        chunk = np.zeros((num_actions + 1, action_dim), dtype=np.float32)
        
        for i in range(num_actions + 1):
            # 限制索引不超过序列长度
            t = min(start_idx + i, T - 1)
            chunk[i] = actions[t]
        
        return chunk


def create_libero_meta(
    
    data_dir: str,
    subsets: List[str] = None,
    output_path: str = None
) -> dict:
    """
    创建LIBERO数据集元配置文件
    
    Args:
        data_dir: LIBERO数据集根目录
        subsets: 要包含的子集列表，默认包含["libero_10", "libero_goal", "libero_object", "libero_spatial"]
        output_path: 保存元JSON的路径（可选）
    
    Returns:
        meta字典，包含数据集基本信息和datalist
    """
    import json
    
    if subsets is None:
        subsets = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]
    
    datalist = []
    
    # 遍历每个子集
    for subset in subsets:
        subset_dir = os.path.join(data_dir, subset)
        if not os.path.exists(subset_dir):
            print(f"警告: {subset_dir} 不存在，跳过")
            continue
            
        # 获取子集下所有HDF5文件
        h5_files = sorted(glob.glob(os.path.join(subset_dir, "*.hdf5")))
        for h5_path in h5_files:
            # 解析任务描述
            base = os.path.basename(h5_path)
            task = re.sub(r"_demo\.hdf5$", "", base)
            m = re.search(r"SCENE\d+_", task)
            if m:
                task = task[m.end():]
            task = task.replace("_", " ")
            
            # 添加到datalist
            datalist.append({
                "path": h5_path,
                "task": task,
                "subset": subset,
            })
    
    # 构建meta字典
    meta = {
        "dataset_name": "libero_hdf5",
        "data_dir": data_dir,
        "datalist": datalist,
        "num_episodes": len(datalist),
        "observation_key": ["obs/agentview_rgb", "obs/eye_in_hand_rgb"],
        "action_key": "actions",
        "state_dim": 8,  # 本体感受维度
        "action_dim": 7,  # 动作维度
        "fps": 10,        # 帧率
    }
    
    # 保存meta到JSON文件
    if output_path:
        with open(output_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"元数据已保存到 {output_path}")
    
    return meta


if __name__ == "__main__":
    # 命令行工具：生成LIBERO元数据
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="LIBERO数据集目录")
    parser.add_argument("--output", type=str, default=None,
                        help="输出元JSON路径")
    args = parser.parse_args()
    
    meta = create_libero_meta(
        args.data_dir,
        output_path=args.output
    )
    print(f"找到 {meta['num_episodes']} 个episode")