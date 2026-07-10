#!/bin/bash
# SimVLA 模型在 LIBERO 数据集上的训练脚本（小模型配置）
# 
# 核心特性：
#   - 384x384 图像分辨率（满足SmolVLM输入要求）
#   - 所有视图由VLM统一处理（无额外辅助视觉输入）
#   - 轻量化动作Transformer配置

set -e  # 遇到错误立即退出脚本

# =============================================================================
# 命令行参数（带默认值）
# =============================================================================

BATCH_SIZE=${1:-64}          # 参数1：批次大小，默认64
LEARNING_COEF=${2:-0.1}      # 参数2：学习率系数，默认0.1
OUTPUT_DIR=${3:-./runs/simvla_libero_small}  # 参数3：输出目录，默认./runs/simvla_libero_small
RESUME_CKPT=${4:-""}        # 参数4：恢复训练的检查点路径，默认空（从头训练）

# 打印训练参数确认
echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"

# GPU配置：使用0-3号GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 抑制TensorFlow日志输出（避免干扰）
export TF_CPP_MIN_LOG_LEVEL=2

# =============================================================================
# 路径配置
# =============================================================================
LIBERO_DATA_DIR="./datasets/metas"          # LIBERO数据集根目录
NORM_STATS_PATH="./norm_stats/libero_norm.json"  # 动作归一化统计文件路径
TRAIN_METAS_PATH="./datasets/metas/libero_train.json"  # 训练元数据文件路径

# SmolVLM骨干模型（支持本地路径或HuggingFace仓库）
SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

# =============================================================================
# 训练超参数
# =============================================================================
LEARNING_RATE=1e-4           # 基础学习率
NUM_ACTIONS=10               # 动作预测窗口长度（action horizon）
ITERS=200000                 # 总训练迭代次数
WARMUP_STEPS=0               # 学习率预热步数（0表示不预热）
FREEZE_STEPS=1000            # 冻结骨干模型的步数（前1000步只训练动作头）
SAVE_INTERVAL=10000          # 检查点保存间隔（每10000步保存一次）
LOG_INTERVAL=20              # 日志打印间隔（每20步打印一次）
NUM_WORKERS=4                # 数据加载线程数
MAX_GRAD_NORM=1.0            # 梯度裁剪的最大范数

# 模型架构配置（小模型）
HIDDEN_SIZE=768              # 隐藏层维度
DEPTH=12                     # Transformer层数
NUM_HEADS=12                 # 注意力头数
USE_ADALN=false              # 是否使用DiT风格的自适应层归一化（默认关闭）

# =============================================================================
# 步骤1：创建训练元数据（如果不存在）
# =============================================================================
if [ ! -f "$TRAIN_METAS_PATH" ]; then
    echo "Creating training metadata..."
    python create_libero_meta.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $TRAIN_METAS_PATH
fi

# =============================================================================
# 步骤2：计算归一化统计信息（如果不存在）
# =============================================================================
if [ ! -f "$NORM_STATS_PATH" ]; then
    echo "Computing normalization statistics..."
    python compute_libero_norm_stats.py \
        --data_dir $LIBERO_DATA_DIR \
        --subsets libero_10 libero_goal libero_object libero_spatial libero_90 \
        --output $NORM_STATS_PATH
fi

# =============================================================================
# 步骤3：构建训练命令参数
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode libero_joint \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --learning_coef ${LEARNING_COEF} \
    --num_actions ${NUM_ACTIONS} \
    --iters ${ITERS} \
    --warmup_steps ${WARMUP_STEPS} \
    --freeze_steps ${FREEZE_STEPS} \
    --hidden_size ${HIDDEN_SIZE} \
    --depth ${DEPTH} \
    --num_heads ${NUM_HEADS} \
    --num_workers ${NUM_WORKERS} \
    --save_interval ${SAVE_INTERVAL} \
    --log_interval ${LOG_INTERVAL} \
    --image_size 384 \
    --norm_stats_path ${NORM_STATS_PATH} \
    --max_grad_norm ${MAX_GRAD_NORM}"

# 如果启用AdaLN，添加对应的命令行参数
if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

# 如果指定了恢复检查点，添加恢复训练的参数
if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# 步骤4：启动训练
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on LIBERO (Small Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Data directory: $LIBERO_DATA_DIR"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: libero_joint"
echo "Batch size: ${BATCH_SIZE}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Learning coef: ${LEARNING_COEF}"
echo "Num actions: ${NUM_ACTIONS}"
echo "Image size: 384x384"
echo "============================================================"
echo "Action Transformer configuration:"
echo "   Hidden size: ${HIDDEN_SIZE}"
echo "   Depth: ${DEPTH}"
echo "   Num heads: ${NUM_HEADS}"
echo "   Use AdaLN: ${USE_ADALN}"
echo "============================================================"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================================"

# 多GPU训练启动命令
# PYTORCH_CUDA_ALLOC_CONF：优化CUDA内存分配（使用可扩展段）
# accelerate launch：多GPU训练工具
#   --num_processes=4：使用4个进程（对应4个GPU）
#   --main_process_port 29504：主进程端口（避免端口冲突）
#   --mixed_precision bf16：使用bf16混合精度训练（加速训练并节省显存）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=4 \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"  # 训练完成提示