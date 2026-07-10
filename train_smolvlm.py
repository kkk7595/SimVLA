"""
SmolVLM-VLA 训练脚本

使用 SmolVLM-500M-Instruct 作为骨干网络训练 SmolVLM-VLA 模型。
使用 512x512 图像分辨率和统一的 VLM 特征（无 aux_visual_inputs）。

使用方法:
    python train_smolvlm.py \
        --output_dir ./runs/smolvlm_vla \
        --train_metas_path ./train_metas.json \
        --batch_size 32 \
        --learning_rate 1e-4 \
        --action_mode galaxea_joint \
        --num_actions 10
"""

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import create_smolvlm_dataloader  # 创建SmolVLM数据加载器
from models.modeling_smolvlm_vla import SmolVLMVLA  # SmolVLM-VLA模型核心类
from models.processing_smolvlm_vla import SmolVLMVLAProcessor  # 数据处理器

import logging
import sys

# WandB 集成（可选）- 用于实验跟踪和可视化
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


# ============================================================
# 日志记录器配置
# ============================================================
def get_logger(name="train_smolvlm", output_dir=None, accelerator=None, level=logging.INFO):
    """
    创建并配置日志记录器
    Args:
        name: 日志器名称
        output_dir: 日志文件输出目录
        accelerator: Accelerator实例，用于判断是否为主进程
        level: 日志级别
    Returns:
        配置好的logger实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 禁止日志传播到父logger
    
    # 如果已有处理器，直接返回
    if logger.handlers:
        return logger
    
    # 判断是否为主进程（分布式训练时只在主进程输出日志）
    is_main = accelerator is None or accelerator.is_main_process
    
    # 日志格式：时间 | 级别 | 名称 | 消息
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"  # 时间格式：小时:分钟:秒
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    
    # 控制台输出处理器（仅主进程）
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    
    # 文件输出处理器（仅主进程）
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train_smolvlm.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    
    return logger


# ============================================================
# 命令行参数解析器
# ============================================================
def get_args_parser():
    """
    创建命令行参数解析器，定义所有训练相关的配置参数
    Returns:
        argparse.ArgumentParser实例
    """
    parser = argparse.ArgumentParser("SmolVLM-VLA 训练脚本", add_help=False)

    # I/O 相关参数
    parser.add_argument("--models", type=str, default=None, 
                        help="预训练SmolVLM-VLA检查点路径（可选）")
    parser.add_argument("--output_dir", type=str, default="runnings_smolvlm", 
                        help="检查点保存目录")

    # SmolVLM 骨干网络参数
    parser.add_argument("--smolvlm_model_path", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM骨干网络的路径或HF仓库名称")
    
    # 数据相关参数
    parser.add_argument("--train_metas_path", type=str, required=True, 
                        help="训练元数据文件路径（必填）")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="训练批次大小")
    parser.add_argument("--image_size", type=int, default=384, 
                        help="SmolVLM输入图像尺寸（默认：384，可选384或512）")

    # 优化器参数
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="基础学习率")
    parser.add_argument("--learning_coef", type=float, default=1.0, 
                        help="VLM骨干网络的学习率乘数")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="权重衰减系数（L2正则）")
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95),
                        help="AdamW优化器的beta参数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="梯度裁剪的最大范数")

    # 训练调度参数
    parser.add_argument("--iters", type=int, default=1000000,
                        help="总训练迭代次数")
    parser.add_argument("--freeze_steps", type=int, default=1000,
                        help="冻结VLM和transformer核心的步数")
    parser.add_argument("--warmup_steps", type=int, default=2000,
                        help="学习率预热步数")
    parser.add_argument("--use_cosine_decay", action="store_true", default=False,
                        help="是否使用余弦退火学习率衰减")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="余弦衰减的最小学习率比例（相对于基础LR）")

    # 日志/保存参数
    parser.add_argument("--save_interval", type=int, default=50000,
                        help="检查点保存间隔（步数）")
    parser.add_argument("--log_interval", type=int, default=20,
                        help="日志输出间隔（步数）")

    # 系统参数
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子，确保实验可复现")
    
    # 动作模式参数
    parser.add_argument("--action_mode", type=str, default="galaxea_joint",
                        help="动作模式：galaxea_joint, galaxea, libero_joint等")
    
    # 数据加载参数
    parser.add_argument("--num_workers", type=int, default=4,
                        help="数据加载的工作进程数")
    
    # 归一化参数
    parser.add_argument("--norm_stats_path", type=str, default=None,
                        help="归一化统计信息的JSON文件路径")
    
    # 动作预测范围
    parser.add_argument("--num_actions", type=int, default=10,
                        help="动作预测的时间范围（预测未来多少个动作）")
    
    # WandB参数
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB项目名称")
    parser.add_argument("--wandb_api_key", type=str, default=None,
                        help="WandB API密钥")
    
    # 恢复训练参数
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从检查点恢复训练")
    
    # DiT/AdaLN模式
    parser.add_argument("--use_adaln", action="store_true", default=False,
                        help="使用DiT风格的AdaLN条件归一化")
    
    # 模型架构参数
    parser.add_argument("--hidden_size", type=int, default=768,
                        help="动作transformer的隐藏层维度")
    parser.add_argument("--depth", type=int, default=12,
                        help="transformer层数")
    parser.add_argument("--num_heads", type=int, default=12,
                        help="注意力头数")

    return parser


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    """
    设置随机种子，确保实验可复现
    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)  # PyTorch种子
    np.random.seed(seed)     # NumPy种子
    random.seed(seed)        # Python原生随机种子
    cudnn.benchmark = True   # 启用CuDNN基准模式，加速训练


def build_optimizer(model: SmolVLMVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_vlm=1.0):
    """
    构建优化器，为不同参数组设置不同的学习率
    Args:
        model: SmolVLMVLA模型实例
        lr: 基础学习率
        weight_decay: 权重衰减
        betas: AdamW的beta参数
        lr_coef_vlm: VLM骨干网络的学习率系数
    Returns:
        配置好的AdamW优化器
    """
    # 分离不同的参数组
    vlm_params = list(model.vlm.parameters())  # VLM骨干网络参数
    
    # 根据模型结构获取动作输出相关参数
    if hasattr(model.transformer, 'final_layer'):
        action_params = list(model.transformer.final_layer.parameters()) + list(model.transformer.action_encoder.parameters())
    else:
        action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    
    # 获取transformer核心参数（排除VLM和动作头）
    exclude = set(map(id, vlm_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    
    # 定义参数组，初始学习率设为0，后续通过调度器更新
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
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


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    """
    获取指定参数组的当前学习率
    Args:
        optim: 优化器实例
        name: 参数组名称
    Returns:
        当前学习率值
    """
    for g in optim.param_groups:
        if g["name"] == name:
            return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
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
    if step < start:
        return 0.0
    
    # 计算预热后的进度
    progress = step - start
    
    # 预热阶段：线性增加学习率
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    
    # 余弦退火阶段
    remain = max(1, total - (start + warmup))  # 剩余步数
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))  # 余弦衰减系数
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)  # 计算最终学习率


def update_group_lrs(optim, step, args):
    """
    更新所有参数组的学习率
    Args:
        optim: 优化器实例
        step: 当前训练步数
        args: 命令行参数
    """
    # 定义各参数组的基础学习率
    base = {
        "vlm": args.learning_rate * args.learning_coef,  # VLM学习率=基础LR×系数
        "transformer_core": args.learning_rate,         # Transformer核心学习率
        "action_heads": args.learning_rate,             # 动作头学习率
    }
    
    # 定义学习率调度函数
    def schedule(step, base_lr):
        return linear_warmup_cosine(
            step, args.freeze_steps, args.warmup_steps, 
            args.iters, base_lr, args.min_lr_ratio
        )
    
    # 冻结阶段：只训练动作头，VLM和Transformer核心学习率设为0
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        # 解冻阶段：根据调度策略更新所有参数组的学习率
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# 主训练流程
# ============================================================
def main(args):
    """
    主训练函数，包含完整的训练流程
    Args:
        args: 解析后的命令行参数
    """
    output_dir = Path(args.output_dir)  # 输出目录路径对象
    
    # WandB配置 - 优先使用环境变量，其次使用命令行参数
    wandb_api_key = os.environ.get("WANDB_API_KEY") or args.wandb_api_key
    wandb_project = os.environ.get("WANDB_PROJECT") or args.wandb_project
    use_wandb = WANDB_AVAILABLE and wandb_api_key  # 判断是否启用WandB

    # 配置日志跟踪器
    log_with = ["tensorboard"]  # 默认使用TensorBoard
    if use_wandb:
        log_with.append("wandb")  # 启用WandB
        os.environ["WANDB_API_KEY"] = wandb_api_key

    # Accelerator配置 - 简化分布式训练、混合精度等设置
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # DDP参数
    accelerator = Accelerator(
        log_with=log_with,
        project_dir=output_dir,
        kwargs_handlers=[ddp_kwargs]
    )

    # 初始化实验跟踪器
    tracker_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "iters": args.iters,
        "smolvlm_model_path": args.smolvlm_model_path,
        "freeze_steps": args.freeze_steps,
        "warmup_steps": args.warmup_steps,
        "save_interval": args.save_interval,
        "action_mode": args.action_mode,
        "num_actions": args.num_actions,
        "image_size": args.image_size,
        "hidden_size": args.hidden_size,
        "depth": args.depth,
        "use_adaln": args.use_adaln,
    }
    
    # 初始化跟踪器
    if use_wandb:
        accelerator.init_trackers(
            project_name=wandb_project,
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"smolvlm-{time.strftime('%Y%m%d-%H%M%S')}"}}
        )
    else:
        accelerator.init_trackers("SmolVLM-VLA-Training", config=tracker_config)

    # 等待所有进程同步
    accelerator.wait_for_everyone()
    # 初始化日志器
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    # 设置随机种子（每个进程使用不同的种子偏移）
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"训练参数: {args}")
    logger.info(f"使用的SmolVLM骨干网络: {args.smolvlm_model_path}")
    logger.info(f"输入图像尺寸: {args.image_size}x{args.image_size}")

    # 加载/初始化模型
    from models.configuration_smolvlm_vla import SmolVLMVLAConfig  # 模型配置类
    from models.action_hub import build_action_space  # 构建动作空间
    
    # 动作空间配置参数
    action_space_kwargs = {}
    if args.norm_stats_path:
        action_space_kwargs["norm_stats_path"] = args.norm_stats_path
        logger.info(f"使用归一化统计信息: {args.norm_stats_path}")
    
    load_path = args.models  # 预训练模型路径
    
    # 从检查点加载模型
    if load_path and os.path.isdir(load_path) and os.path.exists(os.path.join(load_path, "model.safetensors")):
        logger.info(f"从检查点加载SmolVLM-VLA模型: {load_path}")
        model = SmolVLMVLA.from_pretrained(load_path)
        
        # 更新动作模式（如果需要）
        if args.action_mode != model.action_mode:
            logger.warning(f"覆盖模型的action_mode: 从'{model.action_mode}'改为'{args.action_mode}'")
            model.action_mode = args.action_mode
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
        elif action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
            
        # 更新动作预测数量（如果需要）
        if args.num_actions != model.num_actions:
            logger.warning(f"覆盖模型的num_actions: 从{model.num_actions}改为{args.num_actions}")
            model.config.num_actions = args.num_actions
            model.num_actions = args.num_actions
            
        # 检查AdaLN模式（加载后无法更改）
        model_use_adaln = getattr(model, 'use_adaln', False)
        if args.use_adaln != model_use_adaln:
            logger.warning(f"⚠️ 加载检查点后无法更改use_adaln模式")
    else:
        # 从配置初始化新模型
        logger.info(f"从配置初始化SmolVLM-VLA模型")
        logger.info(f"  smolvlm_model_path: {args.smolvlm_model_path}")
        logger.info(f"  action_mode: {args.action_mode}")
        logger.info(f"  num_actions: {args.num_actions}")
        logger.info(f"  use_adaln: {args.use_adaln}")
        
        # 创建模型配置
        config = SmolVLMVLAConfig(
            smolvlm_model_path=args.smolvlm_model_path,
            hidden_size=args.hidden_size,
            depth=args.depth,
            num_heads=args.num_heads,
            action_mode=args.action_mode,
            num_actions=args.num_actions,
            use_adaln=args.use_adaln,
            image_size=args.image_size,
        )
        model = SmolVLMVLA(config)  # 初始化模型
        
        # 设置动作空间（如果需要归一化）
        if action_space_kwargs:
            model.action_space = build_action_space(args.action_mode, **action_space_kwargs)
    
    # 构建数据处理器
    processor = SmolVLMVLAProcessor.from_pretrained(args.smolvlm_model_path)

    # 创建训练数据加载器
    train_dataloader = create_smolvlm_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # 构建优化器
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_vlm=args.learning_coef,
    )
    
    # 使用Accelerator准备模型和优化器（处理分布式、混合精度等）
    model, optim = accelerator.prepare(model, optim)

    # 训练循环
    model.train()  # 设置模型为训练模式
    
    # 恢复训练（如果需要）
    start_step = 0
    if args.resume and load_path and os.path.isdir(load_path):
        state_json = os.path.join(load_path, "state.json")
        if os.path.exists(state_json):
            try:
                with open(state_json, "r") as f:
                    start_step = int(json.load(f).get("global_step", 0))
                logger.info(f"从步数{start_step}恢复训练")
            except Exception as e:
                logger.warning(f"加载训练状态失败: {e}")
    
    # 初始化训练状态
    global_step, t0 = start_step, time.time()
    logger.info(f"🚀 开始SmolVLM-VLA训练，总迭代次数: {args.iters}")
    logger.info(f"   分布式训练进程数: {accelerator.num_processes}")

    # 主训练循环
    for batch in train_dataloader:
        # 编码语言指令
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)  # 移除原始文本，节省内存
        inputs = {**batch, **lang}  # 合并图像和语言输入
        
        # 将所有输入数据移到GPU（非阻塞传输加速）
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        
        # 更新学习率
        update_group_lrs(optim, global_step, args)

        # 前向传播计算损失
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())  # 总损失
        
        # 反向传播
        accelerator.backward(loss)  # 处理分布式反向传播
        # 梯度裁剪，防止梯度爆炸
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()  # 更新参数
        optim.zero_grad()  # 清空梯度

        # 日志记录
        if global_step % args.log_interval == 0:
            # 转换损失值为Python浮点数
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            # 记录各参数组的学习率
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)  # 记录到跟踪器

            # 主进程打印日志
            if accelerator.is_main_process:
                # 计算平均迭代时间
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"总损失={logs['loss_total']:.4f} "
                    f"核心LR={logs['lr_transformer_core']:.2e} "
                    f"动作头LR={logs['lr_action_heads']:.2e} "
                    f"VLM LR={logs['lr_vlm']:.2e} (每步耗时:{dt:.2f}秒)"
                )
        
        # 保存检查点
        global_step += 1  # 更新全局步数
        if accelerator.is_main_process:
            # 达到总步数或保存间隔时保存
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 保存模型到 {save_dir}")
                # 保存模型权重（unwrap去除DDP包装）
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                # 保存训练状态
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
                    
        # 达到总训练步数，终止训练
        if global_step >= args.iters:
            break

    # 结束训练，清理资源
    accelerator.end_training()


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser("SmolVLM-VLA 训练脚本", parents=[get_args_parser()])
    args = parser.parse_args()
    
    # 创建输出目录
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 启动主训练函数
    main(args)

    