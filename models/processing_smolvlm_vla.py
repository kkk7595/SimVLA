"""
SmolVLM-VLA 数据处理器

SmolVLM-VLA 模型的统一多模态数据处理器。
专门处理 SmolVLM-500M 所需的 512x512 图像分辨率。

优化版本特性：
- 基于 GPU 的快速图像预处理
- 缓存归一化参数，避免重复计算
- 最小化 CPU-GPU 数据传输
"""

from transformers import AutoProcessor, AutoTokenizer, AutoImageProcessor
from typing import List, Union, Dict, Any, Optional
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import logging


class SmolVLMVLAProcessor:
    """
    SmolVLMVLAProcessor: SmolVLM-VLA 模型的统一多模态处理器。

    与 FlorenceVLAProcessor 的主要区别：
      - 使用 512x512 图像分辨率（SmolVLM-500M 的要求）
      - 兼容 SmolVLM 的分词器和图像处理器
      - 所有视角图像一起处理，无单独的 aux_visual 处理逻辑

    注意：该类不继承自 ProcessorMixin，以避免分词器类型检查问题。

    属性
    ----------
    num_views : int, 默认=3
        每个样本预期的图像视角数量。
    image_size : int, 默认=384
        SmolVLM 的目标图像尺寸（384x384 或 512x512）。
    language_max_length : int, 默认=50
        文本编码的最大 token 长度。
    """

    num_views: int = 3
    image_size: int = 384
    language_max_length: int = 50

    def __init__(
        self, 
        image_processor=None, 
        tokenizer=None,
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
    ):
        """
        初始化 SmolVLMVLAProcessor 处理器。

        参数
        ----------
        image_processor : PreTrainedImageProcessor, 可选
            SmolVLM 对应的图像处理器。
        tokenizer : PreTrainedTokenizer, 可选
            SmolVLM 对应的文本分词器。
        smolvlm_model_path : str
            用于加载默认处理器的 SmolVLM 模型路径。
        """
        self.smolvlm_model_path = smolvlm_model_path
        
        # 加载 SmolVLM 预训练处理器（包含图像处理器和分词器）
        self._smolvlm_processor = AutoProcessor.from_pretrained(
            smolvlm_model_path,
            trust_remote_code=True,  # 信任远程代码（HF Hub 模型需要）
        )
        
        # 使用提供的处理器或从 SmolVLM 处理器中提取
        self.image_processor = image_processor or self._smolvlm_processor.image_processor
        self.tokenizer = tokenizer or self._smolvlm_processor.tokenizer
        
        # 从图像处理器中获取实际的图像尺寸
        if hasattr(self.image_processor, 'size'):
            size = self.image_processor.size
            if isinstance(size, dict):
                # 处理字典格式的尺寸配置（如 {'height': 384, 'width': 384}）
                self.image_size = size.get('height', size.get('shortest_edge', 384))
            elif isinstance(size, int):
                # 处理整数格式的尺寸配置
                self.image_size = size
        
        # ============ 性能优化：缓存归一化参数 ============
        # 从图像处理器配置中提取均值和标准差
        self._image_mean = None
        self._image_std = None
        if hasattr(self.image_processor, 'image_mean'):
            # 转换为 [1, 3, 1, 1] 形状的张量，方便后续广播运算
            self._image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        if hasattr(self.image_processor, 'image_std'):
            self._image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        
        # 如果未找到归一化参数，使用 ImageNet 标准归一化参数作为默认值
        if self._image_mean is None:
            self._image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        if self._image_std is None:
            self._image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # 记录初始化信息
        logging.info(f"[SmolVLMVLAProcessor] 初始化完成 - 图像尺寸={self.image_size}, "
                     f"归一化均值={self._image_mean.squeeze().tolist()}, 归一化标准差={self._image_std.squeeze().tolist()}")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """
        从预训练的 SmolVLM 模型或本地路径加载处理器。
        
        参数
        ----------
        pretrained_model_name_or_path : str
            Hugging Face 模型名称或本地模型路径
        **kwargs : 其他可选参数
        
        返回
        ----------
        SmolVLMVLAProcessor 实例
        """
        # 优先使用 kwargs 中的路径，否则使用传入的模型路径
        smolvlm_path = kwargs.pop('smolvlm_model_path', pretrained_model_name_or_path)
        
        try:
            # 尝试从指定路径加载
            return cls(smolvlm_model_path=smolvlm_path)
        except Exception as e:
            # 加载失败时使用默认模型路径
            print(f"警告：无法从 {smolvlm_path} 加载处理器: {e}")
            print("使用默认模型：SmolVLM-500M-Instruct")
            return cls(smolvlm_model_path="HuggingFaceTB/SmolVLM-500M-Instruct")

    # ================== 文本编码 ==================
    def encode_language(self, language_instruction: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
        """
        使用 SmolVLM 分词器对语言指令进行 Token 化处理。

        参数
        ----------
        language_instruction : str 或 List[str]
            单个指令文本或一批指令文本。

        返回
        -------
        Dict[str, torch.Tensor]
            {"input_ids": 形状为 [批次大小, 最大长度] 的张量}
        """
        # 统一处理为列表格式（兼容单个文本输入）
        if isinstance(language_instruction, str):
            language_instruction = [language_instruction]

        # 使用分词器进行文本编码
        inputs = self.tokenizer(
            language_instruction,
            return_tensors="pt",          # 返回 PyTorch 张量
            padding="max_length",        # 填充到最大长度
            max_length=self.language_max_length,  # 最大长度限制
            truncation=True,             # 超长文本截断
        )
        # 只返回 input_ids（模型所需的核心文本特征）
        return {"input_ids": inputs["input_ids"]}

    # ================== 优化版图像编码 ==================
    def encode_image(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        为 SmolVLM 预处理多视角图像（384x384 或 512x512）。
        
        优化版本特性：
        - 使用 torch 操作替代 PIL，提升处理速度
        - 将所有图像批量处理，提高效率
        - 避免重复的 CPU-GPU 数据传输

        参数
        ----------
        images : List 或 List[List]
            单样本：[图像1, 图像2, ...]
            批次：[[样本1图像1, 样本1图像2], [样本2图像1, 样本2图像2, 样本2图像3], ...]
            每个图像可以是 PIL.Image、NumPy 数组或 torch.Tensor。

        返回
        -------
        Dict[str, torch.Tensor]
            {
              "image_input": 张量 [批次大小, 视角数, 通道数, 高度, 宽度],
              "image_mask": 张量 [批次大小, 视角数] （标记有效图像视角）
            }
        """
        # 归一化为批次格式（兼容单样本输入）
        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        # 逐样本处理
        for sample_imgs in images:
            processed_tensors = []
            
            # 处理当前样本的所有图像视角
            for img in sample_imgs:
                # 将图像转换为 [通道数, 高度, 宽度] 格式的 float32 张量，值域 [0, 1]
                if isinstance(img, np.ndarray):
                    # NumPy 数组 [H, W, C] -> torch 张量 [C, H, W]
                    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                elif isinstance(img, torch.Tensor):
                    if img.dim() == 3 and img.shape[0] != 3:
                        # 处理 [H, W, C] 格式的张量 -> [C, H, W]
                        tensor = img.permute(2, 0, 1).float()
                    else:
                        tensor = img.float()
                    # 如果值域超过 [0,1]，归一化到 [0,1]
                    if tensor.max() > 1.0:
                        tensor = tensor / 255.0
                elif isinstance(img, Image.Image):
                    # PIL 图像 -> numpy 数组 -> torch 张量
                    np_img = np.array(img)
                    tensor = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0
                else:
                    raise ValueError(f"不支持的图像类型: {type(img)}")
                
                # 使用 torch 快速调整图像大小（双三次插值）
                _, h, w = tensor.shape
                if h != self.image_size or w != self.image_size:
                    tensor = F.interpolate(
                        tensor.unsqueeze(0),  # 添加批次维度 [1, C, H, W]
                        size=(self.image_size, self.image_size),  # 目标尺寸
                        mode='bicubic',       # 双三次插值（高质量）
                        align_corners=False,  # 不对齐角落（避免边缘失真）
                        antialias=True,       # 启用抗锯齿
                    ).squeeze(0)  # 移除批次维度 [C, H, W]
                
                # 使用缓存的均值/标准差进行归一化（广播运算）
                tensor = (tensor - self._image_mean.squeeze(0)) / self._image_std.squeeze(0)
                
                processed_tensors.append(tensor)
            
            # 堆叠当前样本的所有视角图像
            V_exist = len(processed_tensors)  # 实际存在的视角数
            if V_exist > 0:
                processed = torch.stack(processed_tensors, dim=0)  # [视角数, 通道数, 高度, 宽度]
            else:
                # 无图像时创建空张量
                processed = torch.zeros(0, 3, self.image_size, self.image_size)

            # 填充到预设的视角数（不足部分用零填充）
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # 创建图像掩码：有效视角标记为 True，填充视角标记为 False
            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        # 堆叠所有样本，形成批次张量
        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    # ================== 传统（较慢）图像编码 ==================
    def encode_image_legacy(
        self,
        images: Union[List, List[List]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        传统版本 - 保留用于兼容性测试。
        使用 HuggingFace 官方 image_processor（速度较慢但兼容性有保证）。
        """
        # 归一化为批次格式
        if not isinstance(images[0], (list, tuple)):
            images = [images]

        batch_imgs, batch_masks = [], []

        # 逐样本处理
        for sample_imgs in images:
            # 转换为 PIL 图像并调整到目标尺寸
            processed_imgs = []
            for img in sample_imgs:
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                elif isinstance(img, torch.Tensor):
                    if img.dim() == 3:
                        # [C, H, W] -> [H, W, C] 并转换为 uint8
                        img = Image.fromarray(img.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                    else:
                        img = Image.fromarray(img.cpu().numpy().astype(np.uint8))
                
                # 调整图像大小到目标尺寸
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                processed_imgs.append(img)
            
            # 使用 SmolVLM 官方图像处理器进行预处理
            processed = self.image_processor(
                processed_imgs, 
                return_tensors="pt",  # 返回 PyTorch 张量
                **kwargs
            )["pixel_values"]  # 获取处理后的像素值
            
            V_exist = processed.size(0)  # 实际视角数

            # 填充到预设视角数
            if V_exist < self.num_views:
                processed = torch.cat(
                    [processed,
                     processed.new_zeros(self.num_views - V_exist, *processed.shape[1:])],
                    dim=0,
                )

            # 创建图像掩码
            image_mask = torch.zeros(self.num_views, dtype=torch.bool, device=processed.device)
            image_mask[:V_exist] = True

            batch_imgs.append(processed)
            batch_masks.append(image_mask)

        # 堆叠形成批次张量
        image_input = torch.stack(batch_imgs, dim=0)  # [B, num_views, C, H, W]
        image_mask = torch.stack(batch_masks, dim=0)  # [B, num_views]

        return {"image_input": image_input, "image_mask": image_mask}

    # ================== 组合调用 ==================
    def __call__(
        self,
        images: Optional[Union[List, List[List]]] = None,
        language_instruction: Optional[Union[str, List[str]]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        将图像和文本编码组合为统一的多模态输入。

        参数
        ----------
        images : List 或 List[List], 可选
            单样本或批次的多视角图像。
        language_instruction : str 或 List[str], 可选
            对应的文本指令。

        返回
        -------
        Dict[str, torch.Tensor]
            {
              "input_ids": [批次大小, 文本长度],  # 文本Token ID
              "image_input": [批次大小, 视角数, 通道数, 高度, 宽度],  # 图像特征
              "image_mask": [批次大小, 视角数]  # 图像掩码
            }
        """
        outputs: Dict[str, Any] = {}

        # 编码文本（如果提供）
        if language_instruction is not None:
            outputs.update(self.encode_language(language_instruction))

        # 编码图像（如果提供）
        if images is not None:
            outputs.update(self.encode_image(images, **kwargs))

        # 一致性检查：确保文本和图像的批次大小匹配
        if "input_ids" in outputs and "image_input" in outputs:
            assert outputs["input_ids"].size(0) == outputs["image_input"].size(0), (
                f"批次大小不匹配：文本批次 {outputs['input_ids'].size(0)} "
                f"!= 图像批次 {outputs['image_input'].size(0)}"
            )
        return outputs

    def apply_chat_template(
        self,
        images: List[Image.Image],
        text: str,
        add_generation_prompt: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        为多图像输入应用 SmolVLM 的聊天模板。
        
        这在推理阶段非常有用，可以使用 SmolVLM 原生的聊天模板格式。
        
        参数
        ----------
        images : List[Image.Image]
            PIL 图像列表。
        text : str
            文本提示词。
        add_generation_prompt : bool
            是否添加生成提示符（用于后续文本生成）。
            
        返回
        -------
        Dict 包含 input_ids, attention_mask, pixel_values 等字段。
        """
        # 构建聊天内容（图像+文本）
        content = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        
        # 构建符合 SmolVLM 要求的消息格式
        messages = [{"role": "user", "content": content}]
        
        # 应用聊天模板并编码
        inputs = self._smolvlm_processor.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,  # 添加生成提示符
            tokenize=True,  # 执行 Token 化
            return_dict=True,  # 返回字典格式
            return_tensors="pt",  # 返回 PyTorch 张量
        )
        return inputs