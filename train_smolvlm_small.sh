#!/bin/bash
# SimVLA 模型在 DexJoco 数据集上的训练脚本（小模型配置）

set -e  # 遇到错误立即退出脚本

# =============================================================================
# 命令行参数（带默认值）
# =============================================================================
BATCH_SIZE=${1:-64}          
LEARNING_COEF=${2:-0.1}      
OUTPUT_DIR=${3:-./runs/simvla_dexjoco_small}  # 修改：更新了默认输出目录
RESUME_CKPT=${4:-""}        

echo "Training parameters:"
echo "   batch_size: $BATCH_SIZE"
echo "   learning_coef: $LEARNING_COEF"
echo "   output_dir: $OUTPUT_DIR"
echo "   resume_ckpt: ${RESUME_CKPT:-'None (training from scratch)'}"

# GPU配置：请根据你的实际GPU数量修改。如果只有1张卡，写 0
export CUDA_VISIBLE_DEVICES=0

export TF_CPP_MIN_LOG_LEVEL=2
export HF_ENDPOINT=https://hf-mirror.com  # 强制使用国内镜像下载模型

# =============================================================================
# 路径配置 (已修改为 DexJoco 路径)
# =============================================================================
DEXJOCO_DATA_DIR="./datasets"          
NORM_STATS_PATH="./norm_stats/dexjoco_norm.json"  
TRAIN_METAS_PATH="./datasets/metas/dexjoco_train.json"  

SMOLVLM_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

# =============================================================================
# 训练超参数
# =============================================================================
LEARNING_RATE=1e-4           
NUM_ACTIONS=10               
ITERS=200000                 
WARMUP_STEPS=0               
FREEZE_STEPS=1000            
SAVE_INTERVAL=10000          
LOG_INTERVAL=20              
NUM_WORKERS=4                
MAX_GRAD_NORM=1.0            

HIDDEN_SIZE=768              
DEPTH=12                     
NUM_HEADS=12                 
USE_ADALN=false              

# =============================================================================
# 步骤1 & 2：跳过自动生成
# =============================================================================
# ⚠️ 注意：原脚本这里的 create_libero_meta.py 和 compute_libero_norm_stats.py
# 是专门针对 Libero 数据的。由于我们已经手动生成了 dexjoco_train.json 和 dexjoco_norm.json，
# 这里直接将其注释或删除，避免报错。

# =============================================================================
# 步骤3：构建训练命令参数 (已修改 action_mode 和维度、移除不兼容的维度参数)
# =============================================================================
ARGS="--output_dir ${OUTPUT_DIR} \
    --train_metas_path ${TRAIN_METAS_PATH} \
    --smolvlm_model_path ${SMOLVLM_MODEL} \
    --action_mode dexjoco_joint \
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

if [ "${USE_ADALN}" = true ]; then
    ARGS="${ARGS} --use_adaln"
fi

if [ -n "${RESUME_CKPT}" ]; then
    ARGS="${ARGS} --models ${RESUME_CKPT} --resume"
    echo "Resuming from ${RESUME_CKPT}"
fi

# =============================================================================
# 步骤4：启动训练
# =============================================================================
echo "============================================================"
echo "Starting SimVLA Training on DexJoco (Small Action Transformer)"
echo "============================================================"
echo "SmolVLM backbone: ${SMOLVLM_MODEL}"
echo "Normalization stats: $NORM_STATS_PATH"
echo "Action mode: dexjoco_joint"
echo "Action Dim: 44"
echo "============================================================"

# ⚠️ 重要提示：如果你的服务器只有 1 张显卡，请把下面的 --num_processes=4 改为 --num_processes=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch \
    --num_processes=1 \
    --main_process_port 29504 \
    --mixed_precision bf16 \
    train_smolvlm.py ${ARGS}

echo "Training completed!"