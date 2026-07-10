## LIBERO测试

### 复原原作者的结果
```bash
# 基本用法
conda activate libero
python libero_local_eval.py \
    --checkpoint YuankaiLuo/SimVLA-LIBERO \
    --norm_stats ../../norm_stats/libero_norm.json \
    --task_suite libero_spatial \
    --num_trials 50 \
    --gpu_id 0
```

### 测试(保存视频)
```bash
# 基本用法
conda activate libero
python libero_local_eval.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --task_suite libero_spatial \
    --num_trials 50 \
    --gpu_id 0
```

### 测试(不保存视频)
```bash
python libero_local_eval.py \
    --checkpoint ../../runs/simvla_libero_large/ckpt-150000 \
    --norm_stats ../../norm_stats/libero_norm.json \
    --task_suite libero_spatial \
    --num_trials 50 \
    --gpu_id 0 \
    --no_video
```




##
