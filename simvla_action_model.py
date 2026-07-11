"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone.

Key differences from FlorenceVLA:
  - Uses SmolVLM-500M-Instruct (efficient 500M parameter model)
  - 512x512 image input (SmolVLM-500M uses 512x512 patches)
  - All views processed together by SmolVLM, no aux_visual_inputs
  - Unified VLM output for multi-view inputs
"""

from __future__ import annotations

# import json
import json
import logging
import math
import os
import random
# import traceback
from typing import Any, Dict, Literal

import numpy as np
from omegaconf import DictConfig,OmegaConf
import torch
# from torch.optim import AdamW, Optimizer
from fastapi import FastAPI
# from fastapi.responses import JSONResponse
from PIL import Image
# import uvicorn
# import json_numpy
# import cv2
# import jax

import torch.nn.functional as F


from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText

# from rlinf.models.embodiment.simvla.models.modeling_smolvlm_vla import SmolVLMVLA
from rlinf.models.embodiment.simvla.models.processing_smolvlm_vla import SmolVLMVLAProcessor
from rlinf.models.embodiment.simvla.models.transformer_smolvlm import SmolVLMActionTransformer
from rlinf.models.embodiment.simvla.models.action_hub import build_action_space
from rlinf.models.embodiment.simvla.datasets import create_smolvlm_dataloader 

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead

"""
SmolVLM-VLA Configuration

Configuration class for SmolVLM-500M-Instruct based VLA model.
Uses SmolVLM as the vision-language backbone instead of Florence2.
"""

from transformers.configuration_utils import PretrainedConfig
from torch.optim.lr_scheduler import LRScheduler

from transformers import AutoConfig, CONFIG_MAPPING, PretrainedConfig

def SimVLADataloader(model_cfg):
    # ===================== 统一修改 train metas + norm 文件 =====================

    # 1. 从 model_cfg 获取路径（你原项目的配置）
    train_metas_path = model_cfg.train_metas_path
    norm_stats_path = model_cfg.norm_stats_path
    new_data_dir = model_cfg.data_dir

    # ==========================================================================
    # ===================== 1. 修改 train metas JSON =====================
    # ==========================================================================
    with open(train_metas_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 修改顶层 data_dir
    data["data_dir"] = new_data_dir

    # 自动拼接所有 path：{data_dir}/{subset}/{原path}
    for item in data["datalist"]:
        subset = item["subset"]
        old_path = item["path"]
        new_path = os.path.join(new_data_dir, subset, old_path)
        item["path"] = new_path

    # 保存
    with open(train_metas_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # ==========================================================================
    # ===================== 2. 修改 norm JSON =====================
    # ==========================================================================
    with open(norm_stats_path, "r", encoding="utf-8") as f:
        norm_data = json.load(f)

    # 修改 metadata 里的 data_dir
    norm_data["metadata"]["data_dir"] = new_data_dir

    # 保存
    with open(norm_stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_data, f, indent=2, ensure_ascii=False)

    # ===================== 输出结果 =====================
    print("✅ 所有文件路径修改完成！")
    print(f"📁 train_metas: {train_metas_path}")
    print(f"📁 norm_stats : {norm_stats_path}")
    print(f"📍 新 data_dir : {new_data_dir}")

    train_dataloader = create_smolvlm_dataloader(
        batch_size= model_cfg.batch_size,
        metas_path= model_cfg.train_metas_path,
        num_actions= model_cfg.num_actions,
        action_mode= model_cfg.action_mode,
        training=True,
        num_workers= model_cfg.num_workers,
        image_size= model_cfg.image_size,
    )

    return train_dataloader

def SimVLAOptimizer(model,optim_cfg):

    vlm_params = []
    action_params = []
    transformer_core_params = []
    
    # 遍历所有参数，通过参数名进行分组
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过不需要梯度的参数
        
        # 根据参数名特征进行分组
        if "vlm." in name:
            vlm_params.append(param)
        elif "action_expert." in name:
            # 匹配action_expert下的encoder/decoder/final_layer
            if any(key in name for key in ["action_encoder", "action_decoder", "final_layer"]):
                action_params.append(param)
            else:
                transformer_core_params.append(param)
        else:
            transformer_core_params.append(param)

    print(f"VLM骨干网络参数数量: {len(vlm_params)}")
    print(f"动作头参数数量: {len(action_params)}")
    print(f"Transformer核心参数数量: {len(transformer_core_params)}")

    betas = (optim_cfg.adam_beta1, optim_cfg.adam_beta2)
    weight_decay = optim_cfg.get("weight_decay", 1e-2)
    lr = optim_cfg.lr
    
    # 定义参数组
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]

    return torch.optim.AdamW(param_groups, betas=betas)

class SimVLALRScheduler(LRScheduler):
    def __init__(self, 
                config, 
                optim: torch.optim.Optimizer,
    ):
        self.base = {
            "vlm": config.lr * config.learning_coef,  # VLM学习率=基础LR×系数
            "transformer_core": config.lr,         # Transformer核心学习率
            "action_heads": config.lr,             # 动作头学习率
        }

        self.freeze_steps = config.freeze_steps
        self.use_cosine_decay = config.use_cosine_decay
        self.warmup_steps = config.lr_warmup_steps
        self.iters = config.total_training_steps
        self.min_lr_ratio = config.min_lr_ratio

        self.step_count = 0

        self.optim = optim

    def step(self):
        # 定义学习率调度函数
        def schedule(step_count, base_lr):
            return self.linear_warmup_cosine(
                step_count, self.freeze_steps, self.warmup_steps, 
                self.iters, base_lr, self.min_lr_ratio
            )
        
        # 冻结阶段：只训练动作头，VLM和Transformer核心学习率设为0
        if self.step_count < self.freeze_steps:
            self.set_group_lr(self.optim, "vlm", 0.0)
            self.set_group_lr(self.optim, "transformer_core", 0.0)
            self.set_group_lr(self.optim, "action_heads", self.base["action_heads"])
        else:
            # 解冻阶段：根据调度策略更新所有参数组的学习率
            for name, base_lr in self.base.items():
                new_lr = schedule(self.step_count, base_lr) if self.use_cosine_decay else base_lr
                self.set_group_lr(self.optim, name, new_lr)

        self.step_count += 1

    def set_group_lr(self, optim: torch.optim.Optimizer, name: str, lr: float):
        """
        设置指定参数组的学习率
        Args:
            optim: 优化器实例
            name: 参数组名称
            lr: 新的学习率值
        """
        for g in optim.param_groups:
            if g["name"] == name:
                g["lr"] = lr

    def linear_warmup_cosine(self, step_count, start, warmup, total, base_lr, min_ratio):
        """
        实现线性预热+余弦退火的学习率调度
        Args:
            step: 当前训练步数
            start: 开始调度的步数
            warmup: 预热步数
            total: 总训练步数
            base_lr: 基础学习率
            min_ratio: 最小学习率比例
        Returns:
            当前步数的学习率
        """
        # 还未开始调度，返回0
        if step_count < start:
            return 0.0
        
        # 计算预热后的进度
        progress = step_count - start
        
        # 预热阶段：线性增加学习率
        if progress < warmup:
            return base_lr * (progress / max(1, warmup))
        
        # 余弦退火阶段
        remain = max(1, total - (start + warmup))  # 剩余步数
        ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))  # 余弦衰减系数
        return base_lr * (min_ratio + (1 - min_ratio) * ratio)  # 计算最终学习率

class SimVLAConfig(PretrainedConfig):
    """
    Configuration class for the **SmolVLM-VLA (SmolVLM Vision-Language-Action)** model.
    """
    model_type = "simvla"

    def __init__(
        self,
        cfg: DictConfig=None,
        **kwargs  # 必须保留以兼容父类初始化
    ):

        # ========== 基础模型配置 ==========
        self.model_type = cfg.model_type if cfg is not None else "simvla"
        self.model_path = cfg.model_path if cfg is not None else "/path/to/model/simvla"
        self.is_lora = cfg.is_lora if cfg is not None else False

        # ========== SmolVLM核心路径配置 ==========
        self.smolvlm_model_path = cfg.smolvlm_model_path if cfg is not None else "HuggingFaceTB/SmolVLM-500M-Instruct"
        self.train_metas_path = cfg.train_metas_path if cfg is not None else "/kaggle/working/X-RLinf/rlinf/models/embodiment/simvla/datasets/metas/libero_train.json"
        self.norm_stats_path = cfg.norm_stats_path if cfg is not None else "/kaggle/working/X-RLinf/rlinf/models/embodiment/simvla/norm_stats/libero_norm.json"

        # ========== Transformer超参数 ==========
        self.hidden_size = cfg.hidden_size if cfg is not None else 768
        self.depth = cfg.depth if cfg is not None else 12
        self.num_heads = cfg.num_heads if cfg is not None else 12
        self.mlp_ratio = cfg.mlp_ratio if cfg is not None else 4.0
        self.dim_time = cfg.dim_time if cfg is not None else 32
        self.max_len_seq = cfg.max_len_seq if cfg is not None else 512

        # ========== 动作/本体感受配置 ==========
        self.num_actions = cfg.num_actions if cfg is not None else 10
        #kkk
        #self.action_mode = cfg.action_mode if cfg is not None else "libero_joint"
        #训练只跑 DexJoco 数据，可以暂时改成
        self.action_mode = cfg.action_mode if cfg is not None else "dexjoco_joint"
        self.use_proprio = cfg.use_proprio if cfg is not None else True

        # ========== DiT/AdaLN配置 ==========
        self.use_adaln = cfg.use_adaln if cfg is not None else False

        # ========== 图像配置 ==========
        self.image_size = cfg.image_size if cfg is not None else 384
        self.num_views = cfg.num_views if cfg is not None else 3

        # ========== 数据加载配置 ==========
        self.batch_size = cfg.batch_size if cfg is not None else 64
        self.num_workers = cfg.num_workers if cfg is not None else 8

        # ========== RLinf 基础配置 ==========
        self.config_name = cfg.config_name if cfg is not None else "simvla_libero"
        self.num_images_in_input = cfg.num_images_in_input if cfg is not None else 3  # 与num_views默认值一致
        self.noise_method = cfg.noise_method if cfg is not None else "flow_sde"

        # ========== Flow-SDE噪声配置 ==========
        self.noise_level = cfg.noise_level if cfg is not None else 0.5
        self.noise_anneal = cfg.noise_anneal if cfg is not None else False
        self.noise_params = OmegaConf.to_container(cfg.noise_params, resolve=True)  if cfg is not None else [0.7, 0.3, 400]

        # ========== Flow-Noise噪声配置 ==========
        self.noise_logvar_range = OmegaConf.to_container(cfg.noise_logvar_range, resolve=True)  if cfg is not None else [0.08, 0.16]
 
        # ========== 动作相关超参数 ==========
        self.action_chunk = cfg.action_chunk if cfg is not None else 10  # 与num_actions默认值一致
        #kkk
        #self.action_env_dim = cfg.action_env_dim if cfg is not None else 12
        #self.action_dim = cfg.action_dim if cfg is not None else 12  # 补充原配置中遗漏的action_dim
        # 修改: 默认值改为 7 (匹配 DexJoco 旋转向量动作维度)
        self.action_env_dim = cfg.action_env_dim if cfg is not None else 7
        self.action_dim = cfg.action_dim if cfg is not None else 7
        self.num_steps = cfg.num_steps if cfg is not None else 10

        # ========== 训练配置 ==========
        self.train_expert_only = cfg.train_expert_only if cfg is not None else False
        self.safe_get_logprob = cfg.safe_get_logprob if cfg is not None else False
        self.joint_logprob = cfg.joint_logprob if cfg is not None else False
        self.double_layer = cfg.double_layer if cfg is not None else False
        self.ignore_last = cfg.ignore_last if cfg is not None else False

        # ========== 评论家（Critic）配置 ==========
        self.detach_critic_input = cfg.detach_critic_input if cfg is not None else False
        self.chunk_critic_input = cfg.chunk_critic_input if cfg is not None else False
        self.add_value_head = cfg.add_value_head if cfg is not None else False
        self.value_after_vlm = cfg.value_after_vlm if cfg is not None else False
        self.value_vlm_mode = cfg.value_vlm_mode if cfg is not None else "mean_token"

        # 调用父类PretrainedConfig的初始化（必须最后执行）
        super().__init__(**kwargs)

    def to_dict(self):
        """
        Convert this configuration into a fully serializable dictionary.
        """
        output = super().to_dict()
        return output

class SimVLAForRLActionPrediction(PreTrainedModel,BasePolicy):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • SmolVLM-500M-Instruct backbone (vision-language)
      • SmolVLMActionTransformer (flow matching action head)
      • Action space (pre/post-processing + loss)
      
    Key differences from FlorenceVLA:
      • All camera views are input to VLM together (no aux_visual_inputs)
      • 512x512 image resolution (SmolVLM-500M uses 512x512 patches)
      • Efficient 500M parameter model
    """
    config_class = SimVLAConfig   
    base_model_prefix = "simvla"
    supports_gradient_checkpointing = True

    def __init__(self, 
                 config: SimVLAConfig=SimVLAConfig, 
                 *args, 
                 **kwargs
        ):
        super().__init__(config, *args, **kwargs)
        # super().__init__()  # 移除注释，这是关键！

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views

        # smolvlm_path = smolvlm_model_path or "HuggingFaceTB/SmolVLM-500M-Instruct"
        
        # Action space
        action_space_kwargs = {}
        action_space_kwargs["norm_stats_path"] = config.norm_stats_path
        self.action_space = build_action_space(config.action_mode.lower(), **action_space_kwargs)
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # action_env_dim

        # SmolVLM backbone
        smolvlm_model_path =  "HuggingFaceTB/SmolVLM-500M-Instruct"  ## 有问题
        if os.path.isdir(config.smolvlm_model_path):
            smolvlm_model_path = config.smolvlm_model_path
        logging.info(f"Loading SmolVLM from: {smolvlm_model_path}")
        logging.info(f"smolvlm_model_path:   {config.smolvlm_model_path}")

        self.vlm = AutoModelForImageTextToText.from_pretrained(
            smolvlm_model_path,
            torch_dtype=torch.float32,  # Use float32 for training stability
            trust_remote_code=True,
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            smolvlm_model_path,
            trust_remote_code=True,
        )
        self.processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)
        
        # Get SmolVLM hidden size from model config
        # SmolVLM-500M has hidden_size from text_config
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # DiT/AdaLN mode setting
        self.use_adaln = getattr(config, 'use_adaln', False)
        
        # Flow matching action head (SmolVLM version - no aux_visual)
        self.action_expert = SmolVLMActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_adaln=self.use_adaln,
        )
        
        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # Deferred FastAPI app
        self.app: FastAPI | None = None

        
        ############## RLinf ##############
        self.config = config
        
        self.global_step = 0  # 全局训练步数（用于噪声退火）

        # 配置校验：double_layer和joint_logprob不能同时启用
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer和joint_logprob不能同时设置为True"
        )

        # ========== RL专用模块初始化 ==========
        # 初始化值函数头（PPO训练需要）
        if self.config.add_value_head:
            # 创建值函数头（输入→隐藏层→输出1维值）
            self.value_head = ValueHead(
                input_dim=vlm_hidden_size,
                hidden_sizes=(512, 256, 128),   ####
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # 匹配模型参数数据类型（如bf16/fp16）
            self.value_head = self.value_head.to(
                dtype=self.action_expert.weight.dtype
            )

        # 标记是否使用VLM输出计算值函数（Pi05模式）   ##有问题，这部分需要修改
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )

        # 初始化Flow-Noise噪声网络
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=config.hidden_size,                  # 输入维度
                out_dim=self.config.action_env_dim,  # 输出维度（动作维度）  ####
                hidden_dims=[128, 64],        # 隐藏层维度
                activation_type="tanh",       # 激活函数类型
                noise_logvar_range=self.config.noise_logvar_range,  # 噪声范围
                noise_scheduler_type="learn", # 噪声调度类型（可学习）
            )
            # 匹配模型参数数据类型
            self.noise_head = self.noise_head.to(
                dtype=self.action_expert.weight.dtype
            )

        # ========== FSDP分布式适配 ==========
        # 为每个模块设置_fsdp_wrap_name属性（FSDP包装标识）
        for name, module in self.named_modules():  
            path_parts = name.split(".")   
            # 取路径最后一部分作为包装名称（如model.action_in_proj → action_in_proj）
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)  

    def set_global_step(self, global_step):
        """设置全局训练步数（用于噪声退火、学习率调度等）"""
        self.global_step = global_step   

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """
        模型前向传播入口（分发不同模式）
        Args:
            forward_type: 前向传播类型（SFT/DEFAULT）
            **kwargs: 其他参数（如data）
        Returns:
            对应模式的输出（SFT返回损失，DEFAULT返回logprob/value/entropy）
        """
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError(f"不支持的前向传播类型: {forward_type}")

    def sft_forward(self, data, **kwargs):
        """
        SFT（监督微调）前向传播
        Args:
            data: 训练数据字典（含observation/actions）
        Returns:
            loss: SFT训练损失
        """

        """
        yield {
        "language_instruction": instruction,   # 语言指令
        "image_input": image_input,            # 多视图图像输入
        "image_mask": image_mask,              # 图像掩码
        "proprio": torch.tensor(proprio[idx], dtype=torch.float32),  # 本体感受
        "action": torch.tensor(action_chunk, dtype=torch.float32),  # 动作轨迹
        }
        """

        observation = data["observation"]
        action = data["actions"]                                 # [B, T=num_actions, D=dim_action]

       
        image_input = observation["image_input"]     # [B, V, C, H, W]
        image_mask = observation["image_mask"]           # [B, V]
        prompt = observation["input_ids"]        # [B, L] - tokenized language instruction
        state = observation["proprio"]              # [B, dim_proprio]

        enc = self.forward_vlm_efficient(image_input, image_mask, prompt)

        B = prompt.shape[0]
        device = prompt.device
        
        # Beta(1.5, 1) time sampling
        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device), 
            torch.tensor(1.0, device=device)
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001

        # Normalize action and state
        if hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action
            
        if hasattr(self.action_space, 'normalize_state'):
            state_norm = self.action_space.normalize_state(state)
        elif hasattr(self.action_space, 'normalize'):
            state_norm = self.action_space.normalize(state)
        else:
            state_norm = state
        
        # Flow Matching
        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        # Model prediction (no aux_visual_inputs for SmolVLM)
        x, v_t = self.action_expert(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=state_norm,
        )
        
        # MSE loss
        velocity_loss = torch.mean(torch.square(v_t - u_t))
        
        return velocity_loss

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """
        RL默认前向传播（计算logprob/value/entropy）
        核心流程：
        1. 输入预处理→模型输入格式转换
        2. 计算动作对数概率、值函数、熵
        3. 后处理（维度裁剪、均值计算）

        Args:
            data: 输入数据字典（含observation/chains/denoise_inds等）
        Returns:
            dict: 包含logprobs/values/entropy的结果字典
        """

        """
        在EmbodiedFSDPActor里被调用。
        """
        # 解析参数：是否计算值函数
        compute_values = kwargs.get("compute_values", False)

        # print(f"SimVLAForRLActionPrediction.default_forward()  data.keys(): {data.keys()}")
        # dict_keys(['prev_logprobs', 'prev_values', 'dones', 'terminations', 'truncations', 'rewards', 
        # 'chains', 'denoise_inds', 'states', 'image_input', 'image_mask', 'loss_mask', 'loss_mask_sum', 'advantages'])

        chains = data["chains"]          # 动作链（去噪过程的动作序列）
        denoise_inds = data["denoise_inds"]  # 去噪索引
        input_ids: torch.LongTensor = data["input_ids"]        # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor = data["image_input"]     # [B, V, C, H, W]
        image_mask: torch.Tensor = data["image_mask"]           # [B, V]
        state: torch.Tensor = data["states"]              # [B, dim_proprio]
        # action: torch.Tensor = data["actions"]                                 # [B, T=num_actions, D=dim_action]

        # 计算对数概率、值函数、熵
        log_probs, value_t, entropy = self.get_log_prob_value(
            image_input,
            image_mask,
            input_ids,
            state,
            chains,
            denoise_inds,
            compute_values,
        )

        # 维度裁剪：仅保留指定的动作块和环境动作维度
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]

        # 后处理：计算均值，调整维度
        log_probs = log_probs.mean(dim=1)  # 去噪步数维度均值
        # 熵：多维度均值，添加维度以匹配loss-mask形状
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        # 值函数：最后一维均值
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def get_log_prob_value(
        self,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        lang_input,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        """
        计算动作序列的对数概率、值函数和熵
        核心流程：
        1. 计算VLM前缀嵌入和KV缓存
        2. 逐去噪步计算动作均值/标准差→对数概率/熵
        3. 计算值函数（VLM/动作输出）
        4. 整理结果维度

        Args:
            images: 图像张量列表
            img_masks: 图像掩码列表
            lang_tokens: 语言token
            lang_masks: 语言掩码
            state: 状态张量
            chains: 动作链张量
            denoise_inds: 去噪索引
            compute_values: 是否计算值函数
        Returns:
            tuple: (对数概率, 值函数, 熵)
        """
        enc = self.forward_vlm_efficient(image_input, image_mask, lang_input)

        bsize = state.shape[0]  # 批次大小
        D = self.action_space.dim_action
        device = state.device
        dtype = state.dtype

        # Normalize proprio
        if hasattr(self.action_space, 'normalize_state'):
            state_norm = self.action_space.normalize_state(state)
        elif hasattr(self.action_space, 'normalize'):
            state_norm = self.action_space.normalize(state)
        else:
            state_norm = state
        # return self.action_space.postprocess(x_t)

        # 初始化结果列表
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # 联合对数概率模式：计算初始噪声的对数概率和熵
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1  # 非联合模式：仅计算1步

        # 逐步计算对数概率/值函数/熵
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            # 提取当前步和下一步动作
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            
            # 预测动作均值/标准差/值函数
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                enc["vlm_features"],
                chains_pre,  
                state_norm,
                denoise_ind,
                "train",
                self.config.num_steps,
                compute_values,
            )

            # 计算对数概率和熵
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            
            # 记录结果
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            
            # 计算值函数（VLM/动作输出）
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(enc["vlm_features"]))
            else:
                chains_values.append(value_t)

        # 转换为张量（批次×步数×...）
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # 熵处理：仅Flow-Noise模式计算熵，其他模式返回0
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            processed_obs["observation/state"] = env_obs["states"]
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs
    
    def env_obs_processor(self, observation):

        observation.pop("extra_view_images", None)

        # 提取图像并修复维度顺序
        main_images = observation["main_images"]  # 原始维度[B, H, W, C]
        wrist_images = observation["wrist_images"]

        # 将通道维度从最后一维移到第二维（[B, H, W, C] -> [B, C, H, W]）
        if main_images.shape[-1] == 3:  # 确认最后一维是通道数
            main_images = main_images.permute(0, 3, 1, 2)  # 维度重排
            wrist_images = wrist_images.permute(0, 3, 1, 2)

        # 目标尺寸（假设self.config.image_size是单个数值，如224，表示H=W=224）
        target_size = (self.config.image_size, self.config.image_size)
        print(f"Target image size: {target_size}")

        # 检查当前图像尺寸是否与目标尺寸一致，不一致则调整
        # 使用interpolate调整尺寸，保持通道优先格式[B, C, H, W]
        if main_images.shape[2:] != target_size:
            # mode可选：bilinear（双线性插值，适合RGB图）、nearest（最近邻，速度快）
            main_images = F.interpolate(
                main_images, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False  # 避免边缘像素偏移
            )
            wrist_images = F.interpolate(
                wrist_images, 
                size=target_size, 
                mode='bilinear', 
                align_corners=False
            )

        # 合并为image_input [B, V, C, H, W]
        image_input = torch.cat([
            main_images.unsqueeze(1), 
            wrist_images.unsqueeze(1)
        ], dim=1)
        image_input = image_input.float() / 255.0
        observation["image_input"] = image_input

        observation.pop("main_images", None)
        observation.pop("wrist_images", None)

        # 生成image_mask
        observation["image_mask"] = torch.ones(
            (main_images.shape[0], 2), 
            device=main_images.device
        )






        # 4.处理task_descriptions
        lang = self.processor.encode_language(observation["task_descriptions"])
        observation.pop("task_descriptions", None)  # 移除原始文本，节省内存
        observation = {**observation, **lang}  # 合并图像和语言输入

        device = next(self.parameters()).device 
        observation = {k: v.to(device) for k, v in observation.items()}

        return observation

    def input_transform(self, obs: dict, transpose=True):
        """
        first_process     --> prompt
        no first_process  --> tokenized_prompt, tokenized_prompt_mask
        """

        # print(f"SimVLAForRLActionPrediction.input_transform()  type(obs):{type(obs)}")

        # for key in obs.keys():
        #     print(f"SimVLAForRLActionPrediction.input_transform() key: {key}, 对应值类型: {type(obs[key])}")
        
        # inputs = jax.tree.map(lambda x: x, obs)
        # process input
        # first_process = "prompt" in obs.keys()
        # if first_process:
        #     inputs.pop("prompt")
        # else:
        #     inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy
        # inputs = jax.tree.map(
        #     lambda x: np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x, inputs
        # )
        # batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        # split & transform
        # transformed_samples = []
        # for i in range(batch_size):
            # sample = jax.tree.map(lambda x: x[i], inputs)
            # if transpose:
                # convert from [3,256,256] -> [256,256,3]
                # sample = jax.tree.map(
                    # lambda x: x.transpose(1, 2, 0)
                    # if len(x.shape) == 3 and transpose
                    # else x,
                    # sample,
                # )
            # else:
                # sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            # if first_process:
                # sample["prompt"] = obs["prompt"][i]
            # else:
                # sample["prompt"] = "xxxx"

            # transformed_sample = self._input_transform(sample)
            # transformed_samples.append(transformed_sample)
            # transformed_samples.append(sample)
        # recombine
        # inputs = jax.tree.map(
            # lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            # *transformed_samples,
        # )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        # if not first_process:
            # inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            # inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]

        return obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    processed_obs[key][sub_key] = sub_value.to(
                        device=device
                    ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        批量预测动作（RL Rollout核心方法）
        核心流程：
        1. 观测预处理→模型输入变换→精度调整
        2. 动作采样→输出变换→转NumPy
        3. 整理返回结果（动作+logprob/value/输入信息）

        Args:
            env_obs: 环境观测字典
            mode: 模式（train/eval）
            compute_values: 是否计算值函数
            return_obs: 是否返回输入观测
        Returns:
            tuple: (动作数组, 结果字典)

        接口在 EmbodiedFSDPActor.rollout() 中被调用。
        """
        observation = self.env_obs_processor(env_obs)

        # 动作采样（核心逻辑）
        outputs = self.sample_actions(
            observation=observation, mode=mode, compute_values=compute_values
        )

        actions = self.action_space.postprocess(outputs["actions"])
        actions = actions.cpu().numpy()


        # 整理前向输入（用于后续logprob计算）
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }

        forward_inputs.update(observation)
        
        # 整理返回结果
        result = {
            "prev_logprobs": outputs["prev_logprobs"],  # 动作对数概率
            "prev_values": outputs["prev_values"],      # 值函数
            "forward_inputs": forward_inputs,           # 前向输入（复用）
        }

        return actions, result

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
    
    @torch.no_grad()  # 推理阶段禁用梯度计算
    def sample_actions(  
        self,
        observation ,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """
        动作采样核心方法（基于Flow-SDE/Flow-Noise的去噪采样）
        核心流程：
        1. 初始化噪声→预处理观测→计算VLM前缀嵌入
        2. 多步去噪：逐步预测动作均值/标准差→Euler积分更新动作
        3. 计算对数概率/值函数→整理返回结果

        Args:
            observation: 模型观测对象
            noise: 初始噪声（None则自动采样）
            mode: 模式（train/eval）
            compute_values: 是否计算值函数
        Returns:
            dict: 包含actions/chains/prev_logprobs/prev_values/denoise_inds的字典
        """

        # 预处理观测（图像/语言/状态）
        prompt = observation["input_ids"]    # [B, L] - tokenized language instruction
        image_input = observation["image_input"]  # [B, V, C, H, W]
        image_mask = observation["image_mask"]         # [B, V]
        state = observation["states"]  
        
        device = next(self.parameters()).device   # [B, dim_proprio]

        # 计算VLM前缀嵌入和KV缓存
        enc = self.forward_vlm_efficient(image_input, image_mask, prompt)

        bsize = state.shape[0]
        num_steps = self.config.num_steps   # 去噪步数

        # 初始化噪声（默认采样标准正态噪声）
        if noise is None:  
            actions_shape = (bsize, self.config.action_chunk , self.config.action_env_dim)
            noise = self.sample_noise(actions_shape, device)

        # Normalize state            
        if hasattr(self.action_space, 'normalize_state'):
            state_norm = self.action_space.normalize_state(state)
        elif hasattr(self.action_space, 'normalize'):
            state_norm = self.action_space.normalize(state)
        else:
            state_norm = state

        # ========== 去噪采样主循环 ==========
        x_t = noise  # 初始噪声（x_T）
        chains = []  # 存储去噪过程的动作序列
        log_probs = []  # 存储每步对数概率
        values = []     # 存储每步值函数
        chains.append(x_t)  # 记录初始噪声

        # Pi05模式：基于VLM输出计算值函数
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(enc["vlm_features"])
        # 联合对数概率模式：计算初始噪声的对数概率
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # 生成去噪索引（训练/评估模式不同）
        if mode == "train":
            if self.config.joint_logprob:
                # 联合模式：使用所有去噪步数
                denoise_inds = torch.arange(num_steps)
            else:
                # 非联合模式：随机采样一个去噪步（加速训练）
                if self.config.ignore_last:
                    # 忽略最后一步（避免边界效应）
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            # 评估模式：固定索引为-1（使用所有步数）
            denoise_inds = torch.tensor([-1] * num_steps)
        # 扩展到批次维度
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # 逐步去噪
        for idx in range(num_steps):
            # 确定当前步采样模式（train/eval）
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            
            # 预测当前步动作均值、标准差、值函数
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                vlm_features = enc["vlm_features"],
                x_t = x_t,
                state_norm = state_norm,
                idx = idx,
                mode = sample_mode,
                denoise_steps = num_steps,
                compute_values = compute_values,
            ) # 
            
            # Euler积分更新动作：x_{t-1} = mean + noise * std
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            # 计算当前动作的对数概率
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            
            # 记录值函数、动作序列、对数概率
            values.append(value_t)  ##有问题
            chains.append(x_t)
            log_probs.append(log_prob)

        # 最终动作（x_0）
        x_0 = x_t
        # 转换为张量（批次×步数×动作维度）
        chains = torch.stack(chains, dim=1)

        # ========== 后处理 ==========
        # 裁剪对数概率到指定动作维度
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # 对数概率均值计算（联合/非联合模式）
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            # 非联合模式：仅使用采样的去噪步
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]

        # 值函数后处理
        if self.use_vlm_value:
            # Pi05模式：使用VLM值函数（扩展维度）
            values = values_vlm[:, None]
        else:
            # Pi0模式：多步值函数均值
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": x_0,               # 最终采样动作
            "chains": chains,             # 去噪动作序列
            "prev_logprobs": log_probs,   # 动作对数概率
            "prev_values": values,        # 值函数
            "denoise_inds": denoise_inds, # 去噪索引
        }

    def sample_mean_var_val(
        self,
        vlm_features,
        x_t,
        state_norm,
        idx,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        预测指定去噪步的动作均值、标准差和值函数
        核心逻辑：
        1. 计算时间步/噪声强度（支持退火）
        2. 预测动作速度→计算x0/x1预测值
        3. 不同噪声策略计算均值/标准差权重
        4. 计算值函数（如有）

        Args:
            x_t: 当前步动作（x_t）
            idx: 当前去噪步索引
            state: 状态张量
            prefix_pad_masks: 前缀pad掩码
            past_key_values: VLM KV缓存
            mode: 采样模式（train/eval）
            denoise_steps: 总去噪步数
            compute_values: 是否计算值函数
        Returns:
            tuple: (动作均值, 动作标准差, 值函数)
        """
        # 扩展索引到批次维度
        bsize = state_norm.shape[0]
        device = state_norm.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        # ========== 时间/噪声参数计算 ==========
        if self.config.noise_anneal:
            # 噪声退火：根据全局步数调整噪声强度
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # 固定噪声强度
            noise_level = torch.tensor(self.config.noise_level).to(device)

        # 生成时间步（从1到0，包含0）
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])

        # 当前时间步和时间差（delta = t - t'）
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        # ========== 动作速度预测 ==========
        suffix_out, v_t = self.action_expert(
                vlm_features=vlm_features,
                action_with_noise=x_t,
                proprio=state_norm,
                t=t_input,
            )

        # ========== 值函数预测 ==========
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # Pi0模式：基于动作输出计算值函数
            if self.config.chunk_critic_input:
                # 仅使用动作块输入
                print("simvla dont support chunk critic input")

            # 使用全部动作输入
            suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # 解耦评论家输入（避免梯度回传）
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            # 值函数头前向传播
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            # 不计算值函数→返回0
            value_t = torch.zeros((bsize), device=device)

        # ========== ODE-SDE混合采样 ==========
        # 扩展维度以匹配动作形状
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        # 预测x0（t=0）和x1（t=1）的动作值
        x0_pred = x_t - v_t * t_input    # t=0时的动作预测（去噪完成）
        x1_pred = x_t + v_t * (1 - t_input)  # t=1时的动作预测（纯噪声）

        # 不同模式计算均值/标准差权重
        if mode == "eval":
            # 评估模式：无噪声，纯ODE
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)

        elif mode == "train":
            # 训练模式：不同噪声策略
            if self.config.noise_method == "flow_sde":
                # Flow-SDE：基于时间步的标准差
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                # Flow-CPS：余弦相位噪声
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                # Flow-Noise：可学习噪声网络
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out
                )
            else:
                raise ValueError(f"无效的噪声方法: {self.config.noise_method}")

        # 计算最终动作均值（x0和x1加权和）
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_logprob_norm(self, sample, mu, sigma):
        """
        计算正态分布对数概率（支持安全模式）
        公式：log p(x|mu,sigma) = -log(sigma) - 0.5*log(2π) - 0.5*((x-mu)/sigma)²

        Args:
            sample: 采样值（x）
            mu: 均值
            sigma: 标准差
        Returns:
            log_prob: 对数概率张量
        """
        if self.config.safe_get_logprob:
            # 安全模式：仅计算平方项（避免除零/对数错误）
            log_prob = -torch.pow((sample - mu), 2)
        else:
            # 标准模式：完整对数概率计算
            mask = sigma == 0  # 标准差为0的掩码
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)  # 避免除零
            # 常数项：-log(sigma) - 0.5*log(2π)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            # 指数项：-0.5*((x-mu)/sigma)²
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            # 标准差为0时，对数概率设为0
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def get_value_from_vlm(self, prefix_output):
        """
        从VLM输出计算值函数（Pi05模式）
        核心逻辑：
        1. 根据配置选择VLM token（mean/last/first）
        2. 平均选定token→值函数头前向传播

        Args:
            prefix_output: VLM前缀输出 [bs, seq_len, hidden_dim]
        Returns:
            values_vlm: 值函数 [bs]
        """
        all_token_length = self.config.max_len_seq  # 256*3 + 48

        # 根据模式生成token掩码
        if self.config.value_vlm_mode == "mean_token":
            # 平均模式：使用图像token + 语言token
            prefix_mask =  [True] * all_token_length  
        elif self.config.value_vlm_mode == "last_token":
            # 最后token模式：仅使用最后一个token
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            # 首个token模式：仅使用第一个token
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)

        # 提取选定token并平均
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        # 转换为Float32（避免低精度问题）
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        # 值函数头前向传播
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        """
        计算高斯分布的熵
        公式：H = 0.5 * log(2πeσ²)

        Args:
            sigma: 标准差张量
        Returns:
            entropy: 熵张量
        """
        mask = sigma == 0  # 标准差为0的掩码
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)  # 避免除零
        # 熵计算
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        """
        冻结VLM（视觉语言模型）参数（仅训练专家网络）
        核心操作：
        1. 设置VLM为评估模式
        2. 禁用VLM参数梯度计算
        """
        if self.config.train_expert_only:
            print("Do not suprot freeze_vlm now!")
            # self.paligemma_with_expert.paligemma.eval()  # VLM设为评估模式
            # # 禁用VLM所有参数梯度
            # for params in self.paligemma_with_expert.paligemma.parameters():
            #     params.requires_grad = False

    # ============================= SmolVLM encoder =============================
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - multi-view images
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        language_instruction: list[str] | None = None,  # Optional text prompts
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-view images via SmolVLM2.
        
        All views are processed together by SmolVLM, producing unified features.
        No aux_visual_inputs needed - everything goes through VLM.

        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
            
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # Prepare images for SmolVLM - flatten views and filter by mask
        # SmolVLM can handle multiple images as part of multi-image inference
        batch_features = []
        
        for b in range(B):
            # Get valid images for this sample
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]  # [num_valid, C, H, W]
            
            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")
            
            # Convert to PIL images for SmolVLM processor
            pil_images = []
            for img_tensor in valid_images:
                # Denormalize and convert to PIL
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Assuming normalized with ImageNet stats, denormalize
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Build message for SmolVLM with multiple images
            content = []
            for i, img in enumerate(pil_images):
                content.append({"type": "image", "image": img})
            
            # Add text prompt if provided
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})
            
            messages = [{"role": "user", "content": content}]
            
            # Process with SmolVLM
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Get encoder outputs (hidden states) instead of generating text
            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Use the last hidden state as features
            # Shape: [1, seq_len, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))  # [seq_len, hidden_size]
        
        # Pad to same length and stack
        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]
        
        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat
        
        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - Already preprocessed
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - Pre-tokenized text
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient VLM forward for training - uses FULL VLM to fuse vision and language.
        
        Key improvement: Uses complete VLM forward (vision encoder + language model)
        to get features that fuse visual and linguistic information, rather than
        just using the vision encoder alone.
        
        Pipeline:
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return
        
        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # ========== Step 1: Get vision features ==========
        # Flatten images: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        
        # Get valid images
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]
        
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        
        # Encode images through SmolVLM's vision encoder (SigLIP)
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get image features and project to LM space
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]
        
        # Project to language model space using the connector/projector
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)
        
        # ========== Step 2: Get text embeddings ==========
        # Idefics3 (SmolVLM) uses 'text_model' instead of 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # ========== Step 3: Build combined sequence per sample ==========
        # For each sample, concatenate: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        
        # Reconstruct image features with batch structure
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        
        # Count valid views per sample for proper concatenation
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]
        
        batch_inputs_embeds = []
        max_seq_len = 0
        
        for b in range(B):
            # Get valid image features for this sample
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]
            
            # Get text embeddings for this sample
            sample_text_embeds = text_embeds[b]  # [L, D]
            
            # Concatenate: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])
        
        # ========== Step 4: Pad and stack ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1
        
        # ========== Step 5: Forward through text model (Idefics3/SmolVLM) ==========
        # This fuses visual and linguistic information through the full transformer
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use the last hidden state as VLM features
        # This now contains fused vision-language representations
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]
        
        return {"vlm_features": vlm_features}


