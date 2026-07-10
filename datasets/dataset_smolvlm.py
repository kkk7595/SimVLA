"""
SmolVLM 数据集

专为 SmolVLM-VLA 训练设计的数据集类。
与原始数据集的主要区别：
  - 使用 SmolVLM-500M 要求的 512x512 图像分辨率
  - 通过合理的上采样处理小尺寸图像
  - 采用与 SmolVLM 兼容的 ImageNet 归一化参数
"""

from __future__ import annotations
from typing import Dict, Iterable, List
import io
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset  # 导入可迭代数据集基类
from torchvision import transforms
from torchvision.transforms import InterpolationMode  # 插值模式枚举
from mmengine import fileio  # MMEngine 文件IO工具，支持多种存储后端
from .utils import action_slice  # 动作序列切片工具函数
from .domain_config import DATA_WEIGHTS  # 数据集权重配置
from .domain_handler.registry import get_handler_cls  # 数据集处理器注册器


class SmolVLMDataReader(IterableDataset):
    """
    SmolVLM-VLA 训练专用的无限数据读取器。
    
    适配 SmolVLM-500M 要求的 512x512 图像分辨率。
    对小于 512x512 的图像进行合理的上采样处理。
    
    输出样本格式:
      {
        'language_instruction': str,               # 语言指令文本
        'image_input': FloatTensor[V, C, 512, 512], # 图像输入 (V=视角数, C=通道数)
        'image_mask': BoolTensor[V],                # 图像掩码（标记有效视角）
        'proprio': FloatTensor[dim_proprio],        # 本体感知数据
        'action': FloatTensor[T, dim_action],       # 动作序列 (T=预测步数)
      }
    """
    
    # SmolVLM 专用常量配置
    IMAGE_SIZE = 384  # 默认图像尺寸，可调整为 384 或 512
    
    # ImageNet 归一化参数（与 SmolVLM 保持一致）
    IMAGE_MEAN = (0.485, 0.456, 0.406)  # RGB 通道均值
    IMAGE_STD = (0.229, 0.224, 0.225)   # RGB 通道标准差
    
    def __init__(
        self, 
        metas_path: str, 
        num_actions: int = 10, 
        num_views: int = 3, 
        training: bool = True,
        action_mode: str = "galaxea_joint",
        lang_aug: str = None,
        image_size: int = 384,  # 默认 384，支持 384/512
    ):
        """
        初始化 SmolVLM 数据集读取器。
        
        参数
        ----------
        metas_path : str
            元数据文件/目录路径（包含数据集的路径、标签等信息）。
        num_actions : int, 默认=10
            需要预测的未来动作数量（动作预测时间范围）。
        num_views : int, 默认=3
            每个样本的图像视角数量。
        training : bool, 默认=True
            是否为训练模式（训练模式启用数据增强）。
        action_mode : str, 默认="galaxea_joint"
            动作模式（如 "galaxea_joint", "libero_joint"）。
        lang_aug : str, 可选
            语言增强方式（预留参数）。
        image_size : int, 默认=384
            输出图像尺寸（384 或 512）。
        """
        self.num_views = num_views          # 图像视角数
        self.training = training            # 训练/验证模式标记
        self.num_actions = num_actions      # 预测动作数量
        self.action_mode = action_mode      # 动作模式
        self.image_size = image_size        # 图像尺寸
        self.metas: Dict[str, dict] = {}    # 存储各数据集的元数据
        
        # 打印初始化信息
        print(f"[SmolVLM 数据集] 图像尺寸: {self.image_size}x{self.image_size}")
        print(f"[SmolVLM 数据集] 动作模式: {action_mode}")
        
        # 加载元数据
        if fileio.isdir(metas_path):
            # 如果是目录，递归查找所有 json 元数据文件
            meta_files = fileio.list_dir_or_file(
                metas_path, suffix=".json", recursive=True, list_dir=False
            )
            root = metas_path  # 根目录路径
        elif metas_path.endswith('.json'):
            # 如果是单个 json 文件
            try:
                with open(metas_path, 'r') as f:
                    content = json.load(f)
                # 如果文件内容是列表（多个数据集路径）
                if isinstance(content, list):
                    meta_files = content
                    root = ""
                else:
                    meta_files, root = [metas_path], ""
            except Exception:
                # 加载失败时降级为单个文件处理
                meta_files, root = [metas_path], ""
        else:
            # 其他情况视为单个文件
            meta_files, root = [metas_path], ""
            
        # 逐个加载元数据文件
        for file in meta_files:
            # 使用 MMEngine 文件IO读取（支持多种存储后端）
            with io.BytesIO(fileio.get(fileio.join_path(root, file))) as f:
                meta = json.load(f)
            # 打印数据集信息
            print(f"== 加载数据集 {meta['dataset_name']}，包含 {len(meta['datalist'])} 条轨迹")
            self.metas[meta["dataset_name"]] = meta  # 存储元数据

        # 构建适用于 384x384 图像的增强流水线
        self.image_aug = self._build_image_transforms(training)

    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        """
        构建 SmolVLM 专用的图像变换流水线。
        
        处理逻辑：
          - 调整尺寸到 512x512（小图上采样，大图下采样）
          - 训练模式添加色彩抖动增强
          - 应用 ImageNet 归一化
        
        参数
        ----------
        training : bool
            是否为训练模式（决定是否添加数据增强）。
            
        返回
        -------
        transforms.Compose
            组合后的图像变换流水线。
        """
        transform_list = [
            # 调整图像尺寸到目标大小
            # 小图上采样时使用 BICUBIC 插值保证质量
            transforms.Resize(
                (self.image_size, self.image_size), 
                interpolation=InterpolationMode.BICUBIC,  # 双三次插值
                antialias=True,  # 启用抗锯齿
            ),
        ]
        
        # 训练模式添加色彩抖动增强
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,  # 亮度抖动范围
                    contrast=0.2,    # 对比度抖动范围
                    saturation=0.2,  # 饱和度抖动范围
                    hue=0.0          # 色调不抖动（避免颜色失真）
                )
            )
        
        # 转换为张量并应用归一化
        transform_list.extend([
            transforms.ToTensor(),  # PIL/NumPy -> Tensor [C, H, W]，值域 [0,1]
            # 应用 ImageNet 归一化（与 SmolVLM 预训练一致）
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        
        return transforms.Compose(transform_list)  # 组合变换

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        """
        迭代单个数据集的样本。
        
        参数
        ----------
        dataset_name : str
            数据集名称。
            
        返回
        -------
        Iterable[dict]
            样本迭代器。
        """
        meta = self.metas[dataset_name]  # 获取数据集元数据
        traj_indices = list(range(len(meta["datalist"])))  # 轨迹索引列表
        
        # 训练模式下打乱轨迹顺序
        if self.training:
            random.shuffle(traj_indices)

        # print(f"dataset_name：{dataset_name}  meta：{meta}")
            
        # 获取对应数据集的处理器类
        Handler = get_handler_cls(dataset_name)
        # 初始化数据集处理器
        handler = Handler(meta=meta, num_views=self.num_views)

        # 遍历所有轨迹
        for traj_idx in traj_indices:
            try:
                # 迭代轨迹中的每个样本
                print(f"traj_idx：{traj_idx}")
                print(f"num_actions：{self.num_actions}")
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map=meta.get("lang_aug_map"),  # 语言增强映射
                    action_mode=self.action_mode             # 动作模式
                ):
                    # 获取需要计算增量的索引
                    idx_for_delta = meta.get("idx_for_delta", [])
                    # 检查样本是否包含本体感知数据
                    has_proprio = "proprio" in sample
                    # 对动作轨迹进行切片处理（获取指定长度的动作序列）
                    slice_result = action_slice(sample["abs_trajectory"], idx_for_delta)
                    
                    # 更新样本中的动作数据
                    if has_proprio:
                        sample["action"] = slice_result["action"]
                    else:
                        sample.update(slice_result)
                    # 删除原始轨迹数据（节省内存）
                    del sample["abs_trajectory"]
                    
                    # 1. 提取actions（仅保留sample["action"]）
                    actions = sample.pop("action")  # pop同时删除key，避免重复存储
                    
                    # 2. 剩余的sample即为observation（包含除action外的所有键值对）
                    observation = sample  # 此时sample已无action键，直接赋值

                    yield observation, actions

            except Exception as e:
                # 单个轨迹处理失败时跳过，不中断整体训练
                print(f"轨迹 {traj_idx} 处理失败，终止当前数据集迭代: {e}")
                continue
                
        # 训练模式下无限循环（IterableDataset 特性）
        if self.training:
            yield from self._iter_one_dataset(dataset_name)

    def __iter__(self):
        """
        主迭代器入口。
        训练模式：按权重混合多个数据集，无限迭代
        验证模式：按顺序遍历所有数据集，单次迭代
        """
        names = list(self.metas.keys())  # 所有数据集名称
        
        # 验证模式：按顺序遍历
        if not self.training:
            for n in names:
                yield from self._iter_one_dataset(n)
        # 训练模式：按权重随机采样
        else:
            # 创建每个数据集的迭代器
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            # 获取各数据集的权重（默认1.0）
            ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            # 归一化权重
            s = sum(ws)
            ws = [w / s for w in ws]
            
            # 无限循环采样
            while True:
                # 按权重随机选择一个数据集
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                # 产出该数据集的下一个样本
                yield next(gens[i])


class SmolVLMDataReaderWithPadding(SmolVLMDataReader):
    """
    带智能填充的 SmolVLM 数据读取器。
    
    与基础版的区别：
    1. 如果图像小于 512x512，先填充以保持长宽比
    2. 再调整到 512x512 尺寸
    
    这种方式对远小于 512x512 的图像更友好，
    可避免极端上采样导致的失真问题。
    """
    
    # 填充模式配置
    PADDING_MODE = "reflect"  # 可选："constant"（常量填充）、"edge"（边缘填充）
    
    def _build_image_transforms(self, training: bool) -> transforms.Compose:
        """
        构建带智能填充的图像变换流水线。
        
        参数
        ----------
        training : bool
            是否为训练模式。
            
        返回
        -------
        transforms.Compose
            带智能填充的图像变换流水线。
        """
        # 定义智能调整大小的内部类
        class SmartResize:
            """
            智能调整大小类，更好地处理小尺寸图像。
            """
            
            def __init__(self, target_size: int, padding_mode: str = "reflect"):
                """
                初始化智能调整大小处理器。
                
                参数
                ----------
                target_size : int
                    目标图像尺寸。
                padding_mode : str, 默认="reflect"
                    填充模式。
                """
                self.target_size = target_size    # 目标尺寸
                self.padding_mode = padding_mode  # 填充模式
                
            def __call__(self, img):
                """
                对图像进行智能调整大小。
                
                对于远小于目标尺寸的图像：
                - 先填充以保持长宽比
                - 再调整到目标尺寸
                
                参数
                ----------
                img : PIL.Image
                    输入图像。
                    
                返回
                -------
                PIL.Image
                    处理后的图像。
                """
                from PIL import Image
                import numpy as np
                
                w, h = img.size  # 获取图像原始尺寸
                
                # 如果图像两个维度都小于目标尺寸的一半，使用填充策略
                if w < self.target_size // 2 and h < self.target_size // 2:
                    # 创建目标尺寸的空白图像
                    result = Image.new('RGB', (self.target_size, self.target_size))
                    
                    # 计算原始图像的居中位置
                    paste_x = (self.target_size - w) // 2
                    paste_y = (self.target_size - h) // 2
                    # 将原始图像粘贴到中心位置
                    result.paste(img, (paste_x, paste_y))
                    
                    # 转换为 NumPy 数组进行填充处理
                    result_np = np.array(result)
                    
                    # 简单反射填充：复制边界像素
                    if paste_x > 0:
                        # 左侧反射填充
                        result_np[:, :paste_x] = np.flip(
                            result_np[:, paste_x:paste_x*2], axis=1
                        )[:, :paste_x]
                        # 右侧反射填充
                        result_np[:, paste_x+w:] = np.flip(
                            result_np[:, paste_x+w-paste_x:paste_x+w], axis=1
                        )[:, :self.target_size-paste_x-w]
                    
                    return Image.fromarray(result_np)
                else:
                    # 对于尺寸合理的图像，使用标准调整大小
                    return img.resize(
                        (self.target_size, self.target_size),
                        Image.BICUBIC  # 双三次插值
                    )
        
        # 构建变换列表
        transform_list = [
            # 智能调整大小（带填充）
            SmartResize(self.image_size, self.PADDING_MODE),
        ]
        
        # 训练模式添加色彩抖动
        if training:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0
                )
            )
        
        # 转换为张量并归一化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.IMAGE_MEAN, self.IMAGE_STD, inplace=True),
        ])
        
        return transforms.Compose(transform_list)


def worker_init_fn(worker_id: int):
    """
    数据加载工作进程初始化函数。
    
    为每个进程设置独立的随机种子，避免数据增强重复；
    """
    # 生成独立的随机种子
    base_seed = torch.initial_seed() % (2**32)
    import random
    np.random.seed(base_seed)       # NumPy 种子
    random.seed(base_seed)          # Python 原生种子
    torch.manual_seed(base_seed)    # PyTorch 种子
    
def create_smolvlm_dataloader(
    batch_size: int, 
    metas_path: str, 
    num_actions: int,
    training: bool,
    action_mode: str,
    num_workers: int = 4,
    image_size: int = 384,
    use_smart_padding: bool = False,
):
    """
    创建 SmolVLM-VLA 训练专用的数据加载器。
    
    参数
    ----------
    batch_size : int
        训练批次大小。
    metas_path : str
        元数据文件/目录路径。
    num_actions : int
        需要预测的未来动作数量。
    training : bool
        是否为训练模式。
    action_mode : str
        动作模式（如 "galaxea_joint", "libero_joint"）。
    num_workers : int, 默认=4
        数据加载工作进程数。
    image_size : int, 默认=384
        图像尺寸（384 或 512）。
    use_smart_padding : bool, 默认=False
        是否使用智能填充处理小尺寸图像。
        
    返回
    -------
    DataLoader
        配置好的 PyTorch DataLoader。
    """
    from torch.utils.data import DataLoader
    
    # 根据是否使用智能填充选择数据集类
    if use_smart_padding:
        DatasetClass = SmolVLMDataReaderWithPadding  # 带智能填充的数据集
    else:
        DatasetClass = SmolVLMDataReader             # 基础数据集
    
    # 初始化数据集
    dataset = DatasetClass(
        metas_path=metas_path,
        num_actions=num_actions,
        training=training,
        action_mode=action_mode,
        image_size=image_size,
    )
    
    # 创建并返回 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,                # 批次大小
        num_workers=num_workers,              # 工作进程数
        pin_memory=True,                      # 启用内存锁定（加速 GPU 传输）
        worker_init_fn=worker_init_fn,        # 工作进程初始化函数
        persistent_workers=num_workers > 0,   # 保持工作进程常驻（减少开销）
    )

    return dataloader

# 定义模块导出列表（指定可被外部导入的类/函数）
__all__ = [
    "SmolVLMDataReader",               # 基础 SmolVLM 数据集读取类
    "SmolVLMDataReaderWithPadding",    # 带智能填充的数据集读取类
    "create_smolvlm_dataloader",       # 数据加载器创建函数
]

