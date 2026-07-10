#!/usr/bin/env python3
"""
SimVLA LIBERO 本地测评脚本
- 移除服务端-客户端架构，本地直接运行测评
- 单GPU运行，单次仅测评一个任务套件
- 支持指定任务套件、测评次数、GPU ID等参数
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Any

import imageio
import json_numpy
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import torch
from transformers import AutoConfig

# LIBERO 相关导入
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# SimVLA 模型相关（根据实际路径调整）
sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from models.modeling_smolvlm_vla import SmolVLMVLA
    from models.processing_smolvlm_vla import SmolVLMVLAProcessor
    from models.configuration_smolvlm_vla import SimVLAConfig
except ImportError:
    raise ImportError("请确保 SimVLA 模型文件路径正确")

# -----------------------------------------------------------------------------
# 常量配置
# -----------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# 每个任务套件的最大步数
MAX_STEPS = {
    "libero_spatial": 800,
    "libero_object": 800,
    "libero_goal": 800,
    "libero_10": 900,
    "libero_90": 900,
}

NUM_STEPS_WAIT = 10  # 等待物体稳定的步数

# 模型配置
MODEL_CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "image_size": 384,
}

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """四元数转轴角表示（与 robosuite 兼容）"""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def preprocess_images(image0: np.ndarray, image1: np.ndarray, image_size: int) -> tuple:
    """预处理图像为模型输入格式"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    img0 = Image.fromarray(image0.astype(np.uint8))
    img1 = Image.fromarray(image1.astype(np.uint8))
    
    img0_t = transform(img0)
    img1_t = transform(img1)
    
    # 填充到3个视角（模型要求）
    padding = torch.zeros_like(img0_t)
    images = torch.stack([img0_t, img1_t, padding], dim=0)
    image_mask = torch.tensor([[True, True, False]])

    return images.unsqueeze(0), image_mask

# -----------------------------------------------------------------------------
# 本地模型推理类
# -----------------------------------------------------------------------------
class LocalSimVLAModel:
    """本地 SimVLA 模型推理类"""
    def __init__(
        self,
        checkpoint_path: str,
        norm_stats_path: str = None,
        smolvlm_model_path: str = "HuggingFaceTB/SmolVLM-500M-Instruct",
        replan_steps: int = 5,
        device: str = "cuda"
    ):
        self.device = device
        self.replan_steps = replan_steps
        self.action_plan: Deque[np.ndarray] = collections.deque()
        
        # 加载模型和处理器
        print(f"加载 SimVLA 模型: {checkpoint_path}")
        self.model = SmolVLMVLA.from_pretrained(checkpoint_path)
        # self.model = SmolVLMVLA.from_pretrained(checkpoint_path,device_map="cuda:0")

        # # 1. 先加载配置，避免meta设备干扰
        # config = AutoConfig.from_pretrained(checkpoint_path)
        # # 2. 显式指定设备加载模型（匹配你的GPU ID=0）
        # self.model = SmolVLMVLA.from_pretrained(
        #     checkpoint_path,
        #     config=config,
        #     device_map="cuda:0"  # 强制加载到GPU 0
        #     # dtype=torch.float16,  # 替换废弃的 torch_dtype，同时减少显存占用
        #     # ignore_mismatched_sizes=True,  # 可选：避免权重尺寸不匹配的小问题
        # )
        # # 3. 确保模型移到指定设备
        # self.model = self.model.to("cuda:0")

        # # 1. 强制关闭 meta 设备上下文（关键！）
        # if torch.get_default_device() == "meta":
        #     torch.set_default_device(self.device)

        # # 清除所有设备上下文管理器
        # if hasattr(torch, '_C') and hasattr(torch._C, 'default_device'):
        #     torch._C._set_default_device(self.device)
        # # 显式设置默认设备为目标设备（而非 meta）
        # torch.set_default_device(self.device)
        # # 禁用 meta 设备的上下文（兜底）
        # torch._dynamo.config.disable = True  # 避免动态编译干扰设备
        
        # # 2. 先加载配置（仅加载结构，不加载权重，避免meta冲突）
        # # config = AutoConfig.from_pretrained(checkpoint_path)
        # config = SimVLAConfig.from_pretrained(checkpoint_path, device_map=None)  # 直接用自定义配置类加载
        
        # # 3. 显式指定设备加载模型，彻底规避meta设备
        # self.model = SmolVLMVLA.from_pretrained(
        #     checkpoint_path,
        #     config=config,
        #     device_map=None,  # 关键：关闭自动设备映射，避免触发 meta 检测
        #     trust_remote_code=True  # 加载自定义模型必须
        #     # device_map={"": self.device} # 强制所有层加载到指定设备（cuda:0/cpu）
        #     # dtype=torch.float16,  # 替换弃用的 torch_dtype，减少显存占用
        #     # ignore_mismatched_sizes=True,  # 兼容权重尺寸小差异
        #     # torch_dtype=None,  # 显式置空弃用参数，消除警告
        # )
        
        # # 4. 二次确认模型设备（兜底）
        # self.model = self.model.to(self.device)
        # self.model.eval()  # 推理模式

        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.processor = SmolVLMVLAProcessor.from_pretrained(smolvlm_model_path)
        
        # 加载归一化统计信息
        if norm_stats_path and os.path.exists(norm_stats_path):
            print(f"加载归一化统计信息: {norm_stats_path}")
            self.model.action_space.load_norm_stats(norm_stats_path)
        else:
            print("警告：未加载归一化统计信息！")

    def reset(self) -> None:
        """重置动作队列"""
        self.action_plan.clear()

    def infer(self, obs: Dict[str, Any], goal: str) -> np.ndarray:
        """单次推理获取动作"""
        # 提取观测数据
        image0 = obs["image"]
        image1 = obs["wrist_image"]
        state = obs["state"]
        
        # 预处理图像
        images, image_mask = preprocess_images(
            image0, image1, MODEL_CONFIG["image_size"]
        )
        images = images.to(self.device)
        image_mask = image_mask.to(self.device)
        
        # 编码语言指令
        lang = self.processor.encode_language([goal])
        lang = {k: v.to(self.device) for k, v in lang.items()}
        
        # 处理 proprioception
        state = np.pad(state, (0, max(0, MODEL_CONFIG["state_dim"] - len(state))))[:MODEL_CONFIG["state_dim"]]
        proprio_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            actions = self.model.generate_actions(
                input_ids=lang['input_ids'],
                image_input=images,
                image_mask=image_mask,
                proprio=proprio_tensor,
                steps=MODEL_CONFIG["action_horizon"],
            )
        
        return actions.cpu().numpy()[0]

    def step(self, obs: Dict[str, Any], goal: str) -> np.ndarray:
        """获取下一步动作（支持重规划）"""
        if not self.action_plan:
            # 推理获取动作序列
            action_chunk = self.infer(obs, goal)
            
            # 将动作加入队列
            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])
        
        return self.action_plan.popleft()

# -----------------------------------------------------------------------------
# LIBERO 环境初始化
# -----------------------------------------------------------------------------
def get_libero_env(task, resolution: int, seed: int):
    """初始化 LIBERO 环境"""
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# -----------------------------------------------------------------------------
# 核心测评函数
# -----------------------------------------------------------------------------
def eval_libero(
    model: LocalSimVLAModel,
    task_suite_name: str,
    num_trials: int = 50,
    seed: int = 7,
    video_out_path: str = "data/libero/videos",
    save_video: bool = True,
    device: str = "cuda"
) -> float:
    """
    运行单个 LIBERO 任务套件的测评
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 初始化任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)
    
    # 创建视频输出目录
    Path(video_out_path).mkdir(parents=True, exist_ok=True)
    
    # 打印测评信息
    print("=" * 50)
    print(f"LIBERO 测评配置")
    print(f"任务套件: {task_suite_name}")
    print(f"任务数量: {num_tasks}, 每个任务测评次数: {num_trials}")
    print(f"最大步数: {max_steps}, 使用设备: {device}")
    print("=" * 50)
    
    total_episodes = 0
    total_successes = 0
    
    # 遍历所有任务
    for task_id in tqdm(range(num_tasks - 1, -1, -1), desc="任务进度"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)
        
        task_successes = 0

        print(f"Initial states count: {len(initial_states)}")
        
        # 每个任务运行多次测评
        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            # 重置环境和模型
            env.reset()
            model.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])
            
            replay_images = []
            t = 0
            done = False
            
            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    # 前 N 步等待物体稳定
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # 处理图像（旋转180度）
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    if save_video:
                        replay_images.append(img)
                    
                    # 构建状态向量 [eef_pos(3), axis_angle(3), gripper_qpos(2)]
                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])
                    
                    # 封装观测数据
                    obs_dict = {
                        "image": img,
                        "wrist_image": wrist_img,
                        "state": state,
                    }
                    
                    # 获取动作
                    action = model.step(obs_dict, task_description)
                    
                    # 执行动作
                    obs, reward, done, info = env.step(action.tolist())
                    
                    # 任务完成
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    print(f"\n回合 {ep} 出错: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            total_episodes += 1
            
            # 保存视频
            if save_video and replay_images:
                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")[:50]
                video_path = Path(video_out_path) / f"{task_suite_name}_{task_segment}_ep{ep}_{suffix}.mp4"
                imageio.mimwrite(str(video_path), replay_images, fps=10)
            
            # 打印回合结果
            status_icon = "[成功]" if done else "[失败]"
            print(f"\r  {status_icon} 任务 {task_id} 回合 {ep}: {suffix} (步数={t})", end="")
        
        # 关闭环境
        env.close()
        
        # 打印任务统计
        task_success_rate = task_successes / num_trials * 100
        print(f"\n  任务 {task_id} 成功率: {task_successes}/{num_trials} ({task_success_rate:.1f}%)")
    
    # 计算总成功率
    success_rate = total_successes / max(total_episodes, 1) * 100
    print("\n" + "=" * 50)
    print(f"测评完成! 总成功率: {total_successes}/{total_episodes} ({success_rate:.1f}%)")
    print("=" * 50)
    
    return success_rate

# -----------------------------------------------------------------------------
# 主函数
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("LIBERO 本地测评脚本")
    
    # 模型相关参数
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="SimVLA 模型 checkpoint 路径或 HuggingFace repo")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="归一化统计信息 JSON 文件路径")
    parser.add_argument("--smolvlm_model", type=str, 
                        default="HuggingFaceTB/SmolVLM-500M-Instruct",
                        help="SmolVLM 模型路径或 HuggingFace repo")
    parser.add_argument("--replan_steps", type=int, default=5,
                        help="模型重规划步数")
    
    # 测评相关参数
    parser.add_argument("--task_suite", type=str, required=True,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="要测评的任务套件")
    parser.add_argument("--num_trials", type=int, default=50,
                        help="每个任务的测评次数")
    parser.add_argument("--seed", type=int, default=7,
                        help="随机种子")
    parser.add_argument("--video_out", type=str, default="./eval_results",
                        help="视频输出目录")
    parser.add_argument("--no_video", action="store_true",
                        help="禁用视频保存（加速测评）")
    
    # 设备相关参数
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="使用的 GPU ID (默认 0)")
    
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device} (GPU ID: {args.gpu_id})")
    
    # 初始化本地模型
    model = LocalSimVLAModel(
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        smolvlm_model_path=args.smolvlm_model,
        replan_steps=args.replan_steps,
        device=device
    )
    
    # 运行测评
    video_out_path = Path(args.video_out) / args.task_suite
    eval_libero(
        model=model,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        video_out_path=str(video_out_path),
        save_video=not args.no_video,
        device=device
    )

if __name__ == "__main__":
    main()
