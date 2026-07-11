from __future__ import annotations
from typing import Iterable, Tuple, Dict, Type, Optional
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np


# =============================================================================
# Normalization Stats
# =============================================================================
class NormStats:
    """Normalization statistics for action normalization."""
    
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        q01: Optional[np.ndarray] = None,
        q99: Optional[np.ndarray] = None,
    ):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)
        self.q01 = torch.as_tensor(q01, dtype=torch.float32) if q01 is not None else None
        self.q99 = torch.as_tensor(q99, dtype=torch.float32) if q99 is not None else None
        
    def to(self, device):
        """Move to specified device."""
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if self.q01 is not None:
            self.q01 = self.q01.to(device)
        if self.q99 is not None:
            self.q99 = self.q99.to(device)
        return self


def load_norm_stats(path: str) -> Dict[str, NormStats]:
    """
    Load normalization statistics from JSON file.
    
    Supports two formats:
    1. Legacy format (action stats only):
       {"action": {"mean": [...], "std": [...], ...}}
       
    2. Extended format (separate state and actions stats):
       {"norm_stats": {"state": {...}, "actions": {...}}}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found: {path}")
        
    with open(path) as f:
        data = json.load(f)
    
    result = {}
    
    # Check if it's extended format (has norm_stats key)
    if "norm_stats" in data:
        data = data["norm_stats"]
    
    for key, stats in data.items():
        if key == "metadata":
            continue
        result[key] = NormStats(
            mean=np.array(stats["mean"], dtype=np.float32),
            std=np.array(stats["std"], dtype=np.float32),
            q01=np.array(stats.get("q01"), dtype=np.float32) if stats.get("q01") else None,
            q99=np.array(stats.get("q99"), dtype=np.float32) if stats.get("q99") else None,
        )
    return result


# =============================================================================
# Registry
# =============================================================================
ACTION_REGISTRY: Dict[str, Type["BaseActionSpace"]] = {}


def register_action(name: str):
    """Decorator for registering a new action space."""
    def _wrap(cls):
        key = name.lower()
        if key in ACTION_REGISTRY:
            raise KeyError(f"ActionSpace '{key}' already registered -> {ACTION_REGISTRY[key]}")
        ACTION_REGISTRY[key] = cls
        cls.name = key
        return cls
    return _wrap


def build_action_space(name: str, **kwargs) -> "BaseActionSpace":
    """Instantiate a registered action space by name."""
    key = name.lower()
    if key not in ACTION_REGISTRY:
        raise KeyError(f"Unknown action space '{name}'. Available: {list(ACTION_REGISTRY.keys())}")
    return ACTION_REGISTRY[key](**kwargs)


# =============================================================================
# Base class
# =============================================================================
class BaseActionSpace(nn.Module):
    """
    Abstract base class for all action-space definitions.

    Each subclass defines:
      - `dim_action`: dimension of the action vector.
      - `gripper_idx`: indices of gripper channels.
      - `compute_loss(pred, target)`: supervised loss for this space.
      - `preprocess(proprio, action, mode)`: pre-step modifications.
      - `postprocess(action)`: post-step corrections.
    """

    name: str = "base"
    dim_action: int = 0
    gripper_idx: Tuple[int, ...] = ()

    def __init__(self):
        super().__init__()

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Alias for compute_loss."""
        return self.compute_loss(pred, target)

    def preprocess(
        self,
        proprio: torch.Tensor,
        action: torch.Tensor,
        mode: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Default: return unchanged."""
        return proprio, action

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Default: return unchanged."""
        return action


# =============================================================================
# Utilities
# =============================================================================
def _ensure_indices_valid(D: int, idx: Iterable[int], name: str) -> None:
    bad = [i for i in idx if i < 0 or i >= D]
    if bad:
        raise IndexError(f"{name} contains out-of-range indices {bad} for action dim D={D}")


# =============================================================================
# LIBERO Action Space
# =============================================================================
@register_action("libero_joint")
class LiberoJointActionSpace(BaseActionSpace):
    """
    LIBERO joint/delta action space.
    
    Data layout:
      - state (proprio): 8-dim [ee_pos(3), ee_ori(3), gripper_states(2)]
      - actions: 7-dim [delta_xyz(3), delta_euler(3), gripper_cmd(1)]
      
    Actions range: [-1, 1] (normalized delta actions)
    
    - Uses MSE loss
    - Optional Z-score or Quantile normalization
    """

    dim_action = 7
    dim_proprio = 8
    gripper_idx = (6,)  # Last dimension is gripper

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
    ):
        super().__init__()
        self.use_quantile_norm = use_quantile_norm
        self.state_norm_stats: Optional[NormStats] = None
        self.action_norm_stats: Optional[NormStats] = None
        
        if norm_stats_path:
            self.load_norm_stats(norm_stats_path)
            
    def load_norm_stats(self, path: str):
        """Load normalization statistics."""
        stats_dict = load_norm_stats(path)
        
        if "state" in stats_dict:
            self.state_norm_stats = stats_dict["state"]
            print(f"[LiberoJointActionSpace] Loaded state norm stats, dim={len(self.state_norm_stats.mean)}")
            
        if "actions" in stats_dict:
            self.action_norm_stats = stats_dict["actions"]
            print(f"[LiberoJointActionSpace] Loaded actions norm stats, dim={len(self.action_norm_stats.mean)}")
            
    def to(self, device):
        """Move to specified device."""
        if self.state_norm_stats is not None:
            self.state_norm_stats.to(device)
        if self.action_norm_stats is not None:
            self.action_norm_stats.to(device)
        return super().to(device)
    
    def _normalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Normalize using specified statistics."""
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        D = x.shape[-1]
        
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return (x - mean) / (std + 1e-6)
    
    def _unnormalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """Unnormalize using specified statistics."""
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        D = x.shape[-1]
            
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return x * (std + 1e-6) + mean
    
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state/proprio."""
        if self.state_norm_stats is not None:
            return self._normalize_with_stats(x, self.state_norm_stats)
        return x
    
    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize action."""
        if self.action_norm_stats is not None:
            return self._normalize_with_stats(x, self.action_norm_stats)
        return x
    
    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """Unnormalize action."""
        if self.action_norm_stats is not None:
            return self._unnormalize_with_stats(x, self.action_norm_stats)
        return x

    def compute_loss(self, pred, target):
        """Full-dimension MSE loss."""
        loss = torch.square(pred - target)
        return {"velocity_loss": torch.mean(loss)}

    def preprocess(self, proprio, action, mode="train"):
        """Normalize proprio and action separately."""
        proprio_norm = self.normalize_state(proprio)
        action_norm = self.normalize_action(action)
        return proprio_norm, action_norm

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """Unnormalize action."""
        return self.unnormalize_action(action)
    


@register_action("franka_joint")
class FrankaJointActionSpace(BaseActionSpace):
    """
    LIBERO 关节/增量动作空间类
    
    数据结构说明:
      - 状态(本体感知): 12维 [末端执行器位置(3), 末端执行器姿态(3), 灵巧手关节角(6)]
      - 动作: 12维 [位置增量xyz(3), 欧拉角增量(3), 灵巧手关节角(6)]
      
    动作数值范围: [-1, 1] (归一化后的增量动作)
    
    - 损失函数: 均方误差损失(MSE)
    - 支持: Z-score标准化 或 分位数归一化(可选)
    """

    # 动作维度：7维
    dim_action = 12
    # 本体感知状态维度：8维
    dim_proprio = 12
    # # 夹爪控制对应的索引：最后一维(第6位)
    # gripper_idx = (6,)

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
    ):
        # 调用父类构造函数
        super().__init__()
        # 是否使用分位数归一化
        self.use_quantile_norm = use_quantile_norm
        # 状态归一化统计参数（均值、标准差、分位数等）
        self.state_norm_stats: Optional[NormStats] = None
        # 动作归一化统计参数
        self.action_norm_stats: Optional[NormStats] = None
        
        # 如果传入了归一化统计文件路径，加载参数
        if norm_stats_path:
            self.load_norm_stats(norm_stats_path)
            
    def load_norm_stats(self, path: str):
        """加载归一化统计参数"""
        stats_dict = load_norm_stats(path)
        
        # 加载状态的归一化参数
        if "state" in stats_dict:
            self.state_norm_stats = stats_dict["state"]
            print(f"[LiberoJointActionSpace] 已加载状态归一化参数，维度={len(self.state_norm_stats.mean)}")
            
        # 加载动作的归一化参数
        if "actions" in stats_dict:
            self.action_norm_stats = stats_dict["actions"]
            print(f"[LiberoJointActionSpace] 已加载动作归一化参数，维度={len(self.action_norm_stats.mean)}")
            
    def to(self, device):
        """将张量移动到指定设备(CPU/GPU)"""
        if self.state_norm_stats is not None:
            self.state_norm_stats.to(device)
        if self.action_norm_stats is not None:
            self.action_norm_stats.to(device)
        return super().to(device)
    
    def _normalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """使用指定的统计参数对数据进行归一化（内部工具方法）"""
        # 确保统计参数和输入数据在同一设备上
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        # 获取输入数据的最后一维维度
        D = x.shape[-1]
        
        # 分位数归一化：将数据缩放到 [-1, 1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        # Z-score标准化：(x-均值)/标准差
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return (x - mean) / (std + 1e-6)
    
    def _unnormalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        """使用指定的统计参数对数据进行反归一化（内部工具方法）"""
        # 设备对齐
        if stats.mean.device != x.device:
            stats.to(x.device)
        
        D = x.shape[-1]
            
        # 分位数反归一化
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01 = stats.q01[..., :D]
            q99 = stats.q99[..., :D]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        # Z-score反标准化
        else:
            mean = stats.mean[..., :D]
            std = stats.std[..., :D]
            return x * (std + 1e-6) + mean
    
    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """对状态/本体感知数据进行归一化"""
        if self.state_norm_stats is not None:
            return self._normalize_with_stats(x, self.state_norm_stats)
        return x
    
    def normalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """对动作数据进行归一化"""
        if self.action_norm_stats is not None:
            return self._normalize_with_stats(x, self.action_norm_stats)
        return x
    
    def unnormalize_action(self, x: torch.Tensor) -> torch.Tensor:
        """对动作数据进行反归一化（还原真实动作值）"""
        if self.action_norm_stats is not None:
            return self._unnormalize_with_stats(x, self.action_norm_stats)
        return x

    def compute_loss(self, pred, target):
        """计算全维度均方误差损失(MSE)"""
        loss = torch.square(pred - target)
        # 返回损失字典，键为velocity_loss
        return {"velocity_loss": torch.mean(loss)}

    def preprocess(self, proprio, action, mode="train"):
        """数据预处理：分别归一化本体感知数据和动作数据"""
        proprio_norm = self.normalize_state(proprio)
        action_norm = self.normalize_action(action)
        return proprio_norm, action_norm

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """数据后处理：对模型输出的动作进行反归一化"""
        return self.unnormalize_action(action)

#kkk
# =============================================================================
# DexJoco Action Space (Rotation Vector)
# =============================================================================
@register_action("dexjoco_joint")
class DexJocoJointActionSpace(BaseActionSpace):
    """
    DexJoco 动作空间类 (将轴角替换为旋转向量)
    
    数据结构说明:
      - 状态(本体感知): 假设 7 维 [位置(3), 旋转向量(3), 夹爪(1)] (请根据学长数据微调)
      - 动作: 7 维 [位置(3), 旋转向量(3), 夹爪(1)]
    """
    # 动作维度：从 8维 变成 7维 (旋转向量为 3维)
    dim_action = 7
    # 状态维度：根据实际 DexJoco 采集的数据维度调整，这里先写 7
    dim_proprio = 7
    
    # 夹爪索引
    gripper_idx = (6,)

    def __init__(
        self,
        norm_stats_path: Optional[str] = None,
        use_quantile_norm: bool = False,
    ):
        super().__init__()
        self.use_quantile_norm = use_quantile_norm
        self.state_norm_stats: Optional[NormStats] = None
        self.action_norm_stats: Optional[NormStats] = None
        
        if norm_stats_path:
            self.load_norm_stats(norm_stats_path)
            
    # ------ 以下代码可以直接复用 FrankaJointActionSpace 的归一化逻辑 ------
    def load_norm_stats(self, path: str):
        stats_dict = load_norm_stats(path)
        if "state" in stats_dict:
            self.state_norm_stats = stats_dict["state"]
        if "actions" in stats_dict:
            self.action_norm_stats = stats_dict["actions"]
            
    def to(self, device):
        if self.state_norm_stats is not None: self.state_norm_stats.to(device)
        if self.action_norm_stats is not None: self.action_norm_stats.to(device)
        return super().to(device)
    
    def _normalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if stats.mean.device != x.device: stats.to(x.device)
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01, q99 = stats.q01[..., :D], stats.q99[..., :D]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            mean, std = stats.mean[..., :D], stats.std[..., :D]
            return (x - mean) / (std + 1e-6)
    
    def _unnormalize_with_stats(self, x: torch.Tensor, stats: NormStats) -> torch.Tensor:
        if stats.mean.device != x.device: stats.to(x.device)
        D = x.shape[-1]
        if self.use_quantile_norm and stats.q01 is not None and stats.q99 is not None:
            q01, q99 = stats.q01[..., :D], stats.q99[..., :D]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            mean, std = stats.mean[..., :D], stats.std[..., :D]
            return x * (std + 1e-6) + mean
    
    def normalize_state(self, x: torch.Tensor): return self._normalize_with_stats(x, self.state_norm_stats) if self.state_norm_stats else x
    def normalize_action(self, x: torch.Tensor): return self._normalize_with_stats(x, self.action_norm_stats) if self.action_norm_stats else x
    def unnormalize_action(self, x: torch.Tensor): return self._unnormalize_with_stats(x, self.action_norm_stats) if self.action_norm_stats else x
    
    def compute_loss(self, pred, target):
        return {"velocity_loss": torch.mean(torch.square(pred - target))}

    def preprocess(self, proprio, action, mode="train"):
        return self.normalize_state(proprio), self.normalize_action(action)

    def postprocess(self, action: torch.Tensor):
        return self.unnormalize_action(action)
    

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    "BaseActionSpace",
    "build_action_space",
    "register_action",
    "LiberoJointActionSpace",
    "ACTION_REGISTRY",
    "NormStats",
    "load_norm_stats",
]
