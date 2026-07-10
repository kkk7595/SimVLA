from __future__ import annotations

import io
import random
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Optional, Sequence, Any

import numpy as np
import h5py
import torch
from mmengine import fileio  # 兼容本地/远程文件读取
from PIL import Image
from scipy.interpolate import interp1d  # 一维插值


class DomainHandler(ABC):
    """
    领域处理器抽象接口（Domain Handler）
    最小化的领域处理接口，子类需实现迭代器方法，生成与训练循环兼容的样本字典
    
    核心作用：
    - 为不同数据集提供统一的解码接口
    - 生成训练/评估所需的样本（包含图像、语言指令、动作轨迹等）
    """
    dataset_name: str  # 数据集名称（子类需定义）

    def __init__(self, meta: dict, num_views: int) -> None:
        """
        初始化领域处理器
        Args:
            meta: 数据集元信息字典（包含观测键、语言指令键、数据列表等）
            num_views: 多视图图像数量（如主视图/辅助视图）
        """
        self.meta = meta
        self.num_views = num_views

    @abstractmethod
    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        action_mode,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """
        生成单个episode（轨迹）的样本迭代器
        子类必须实现此方法，返回可迭代的样本字典
        
        Args:
            traj_idx: 轨迹索引（对应datalist中的第traj_idx个文件）
            num_actions: 每个样本的动作序列长度
            training: 是否为训练模式（影响数据增强、采样策略）
            image_aug: 图像增强函数/变换
            action_mode: 动作模式（如ee6d/joint7）
            lang_aug_map: 语言增强映射表（指令→增强指令列表）
            **kwargs: 其他自定义参数
        
        Yields:
            dict: 单个样本字典，包含语言指令、图像、动作轨迹等
        """
        ...


def _open_h5(path: str) -> h5py.File:
    """
    打开HDF5文件（兼容本地文件系统和远程存储后端）
    通过mmengine.fileio支持多种后端（如本地、S3、HTTP等）
    
    Args:
        path: HDF5文件路径（本地路径或远程URL）
    
    Returns:
        h5py.File: 打开的HDF5文件对象
    
    Raises:
        OSError: 本地打开失败时，尝试通过mmengine读取字节流后打开
    """
    try:
        # 优先本地打开
        return h5py.File(path, "r")
    except OSError:
        # 本地打开失败，通过mmengine读取字节流（兼容远程文件）
        return h5py.File(io.BytesIO(fileio.get(path)), "r")


class BaseHDF5Handler(DomainHandler):
    """
    通用HDF5数据集处理器（资源安全的迭代器）
    基础HDF5处理类，提供通用的HDF5数据集读取和样本生成逻辑，子类仅需实现：
      1. build_left_right(f)：构建左右轨迹、时间轴、频率和未来窗口时长
      2. index_candidates(T_left, training)：生成候选采样索引
    
    可选重写：
      - get_image_datasets(f)：获取图像数据集列表
      - read_instruction(f)：读取语言指令
    """

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        """
        获取图像数据集列表（默认实现）
        从HDF5文件中读取指定观测键对应的图像数据
        
        Args:
            f: 打开的HDF5文件对象
        
        Returns:
            Sequence[Any]: 图像数据集列表，每个元素为图像数组/字节流
        """
        # 获取元信息中的观测键（如["image_main", "image_aux"]）
        keys: Sequence[str] = self.meta["observation_key"]
        # 读取每个键对应的数据集并转换为numpy数组
        return [f[k][()] for k in keys]

    def read_instruction(self, f: h5py.File) -> str:
        """
        读取语言指令（默认实现）
        从HDF5文件中读取语言指令键对应的文本
        
        Args:
            f: 打开的HDF5文件对象
        
        Returns:
            str: 解码后的语言指令字符串
        """
        # 获取元信息中的语言指令键（如"language_instruction"）
        key: str = self.meta["language_instruction_key"]
        ds = f[key]
        v = ds[()]
        # 处理字节字符串解码：标量→直接解码，数组→取第一个元素解码
        return v.decode() if getattr(ds, "shape", ()) == () else v[0].decode()

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        """
        构建左右轨迹、时间轴、频率和未来窗口时长（子类必须实现）
        核心方法，解析HDF5文件中的轨迹数据
        
        Args:
            f: 打开的HDF5文件对象
        
        Returns:
            Tuple: (left轨迹, right轨迹, left时间轴, right时间轴, 频率(Hz), 未来窗口时长(秒))
            - left/right: 绝对轨迹数组 [T, C]
            - left_time/right_time: 可选时间数组 [T]（None则使用默认时间轴）
            - freq: 采样频率（Hz）
            - qdur: 未来预测窗口时长（秒）
        """
        raise NotImplementedError

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        """
        生成候选采样索引（子类必须实现）
        根据轨迹长度和训练模式，生成可采样的索引列表
        
        Args:
            T_left: left轨迹的长度
            training: 是否为训练模式（影响索引生成策略）
        
        Returns:
            Iterable[int]: 候选索引迭代器
        """
        raise NotImplementedError

    @staticmethod
    def _pil_from_arr(arr: Any) -> Image.Image:
        """
        将数组/字节流转换为PIL图像
        兼容不同存储格式的图像数据
        
        Args:
            arr: 图像数组或字节流
        
        Returns:
            Image.Image: PIL图像对象
        """
        from ..utils import decode_image_from_bytes  # 图像解码工具函数
        # 字节流→解码为PIL图像，已为PIL图像则直接返回
        return decode_image_from_bytes(arr) if not isinstance(arr, Image.Image) else arr

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs
    ) -> Iterable[dict]:
        """
        生成单个episode的样本迭代器（核心实现）
        安全打开HDF5文件，生成多个样本，退出时确保文件关闭
        
        核心流程：
        1. 打开HDF5文件，读取图像、语言指令、轨迹数据
        2. 构建时间轴和插值函数
        3. 生成候选采样索引（训练时随机打乱）
        4. 对每个索引，插值生成动作序列，处理图像和语言指令
        5. 生成样本字典并yield
        
        Args:
            traj_idx: 轨迹索引
            num_actions: 动作序列长度
            training: 是否训练模式
            image_aug: 图像增强函数
            lang_aug_map: 语言增强映射表
            **kwargs: 其他参数
        
        Yields:
            dict: 样本字典，包含以下键：
                - language_instruction: 语言指令
                - image_input: 多视图图像张量 [num_views, C, H, W]
                - image_mask: 图像掩码（标记有效视图）[num_views]
                - abs_trajectory: 绝对轨迹张量 [num_actions+1, 2*C]
        """
        # 步骤1：获取轨迹文件路径
        datapath = self.meta["datalist"][traj_idx]
        # 兼容datalist中路径为列表的情况（取第一个）
        if not isinstance(datapath, str):
            datapath = datapath[0]

        # 步骤2：安全打开HDF5文件（with语句确保文件关闭）
        with _open_h5(datapath) as f:
            # 读取图像数据集
            images = self.get_image_datasets(f)
            # 读取语言指令
            ins = self.read_instruction(f)
            # 构建轨迹、时间轴、频率和窗口时长（子类实现）
            left, right, lt, rt, freq, qdur = self.build_left_right(f)
        
        # 步骤3：构建图像掩码（标记有效视图，不足补False）
        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[:len(images)] = True  # 有效视图标记为True
        
        # 步骤4：构建时间轴（None则使用默认时间轴：索引/频率）
        if lt is None: 
            lt = np.arange(left.shape[0], dtype=np.float64) / float(freq)
        if rt is None: 
            rt = np.arange(right.shape[0], dtype=np.float64) / float(freq)

        # 步骤5：生成候选采样索引（子类实现）
        idxs = list(self.index_candidates(left.shape[0], training))
        # 训练模式下随机打乱索引（增强数据随机性）
        if training: random.shuffle(idxs)

        # 步骤6：构建插值函数（用于生成等间隔动作序列）
        # L/R：基于时间轴的轨迹插值函数，超出边界时填充首尾值
        L = interp1d(lt, left, axis=0, bounds_error=False, fill_value=(left[0], left[-1]))
        R = interp1d(rt, right, axis=0, bounds_error=False, fill_value=(right[0], right[-1]))
        # 参考时间轴（左右时间轴的均值）
        ref = (lt + rt) / 2.0

        # 步骤7：确定有效视图数量（不超过设定的num_views）
        V = min(self.num_views, len(images))

        # 步骤8：遍历索引生成样本
        for idx in idxs:
            # 当前时间点
            cur = ref[idx]
            # 生成查询时间点：从cur到cur+qdur，共num_actions+1个点（包含起点和终点）
            q = np.linspace(cur, min(cur + qdur, float(ref.max())), num_actions + 1, dtype=np.float32)
            
            # 插值生成轨迹序列
            lseq = torch.tensor(L(q))  # left轨迹序列 [num_actions+1, C]
            rseq = torch.tensor(R(q))  # right轨迹序列 [num_actions+1, C]

            # 过滤无效样本（轨迹无变化）
            if (lseq[1] - lseq[0]).abs().max() < 1e-5 and (rseq[1] - rseq[0]).abs().max() < 1e-5:
                continue
            
            # 语言增强（训练模式且有增强映射表）
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])
            
            # 处理多视图图像：
            # 1. 对有效视图应用图像增强
            # 2. 不足num_views的视图补零张量
            imgs = [image_aug(self._pil_from_arr(images[v][idx])) for v in range(V)]
            while len(imgs) < self.num_views: 
                imgs.append(torch.zeros_like(imgs[0]))
            # 堆叠为多视图图像张量 [num_views, C, H, W]
            image_input = torch.stack(imgs, dim=0)

            # 生成样本字典并yield
            yield {
                "language_instruction": ins,          # 语言指令
                "image_input": image_input,            # 多视图图像输入
                "image_mask": image_mask,              # 图像掩码（有效视图标记）
                "abs_trajectory": torch.cat([lseq, rseq], -1).float()  # 拼接左右轨迹 [num_actions+1, 2*C]
            }