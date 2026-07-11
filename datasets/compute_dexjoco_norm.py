#kkk
import h5py
import numpy as np
import json
import os

def compute_norm_stats(h5_file_path, output_json_path):
    print(f"⏳ 正在读取数据集计算统计量: {h5_file_path}")
    
    with h5py.File(h5_file_path, 'r') as f:
        # 读取全部的状态和动作数据
        states = f['data/obs/robot0_proprio'][:]
        actions = f['data/actions'][:]
        
    print(f"📊 提取到状态维度: {states.shape}, 动作维度: {actions.shape}")
    
    # 计算均值、标准差、分位数 (1% 和 99%)
    stats = {
        "norm_stats": {
            "state": {
                "mean": np.mean(states, axis=0).tolist(),
                "std": np.std(states, axis=0).tolist(),
                "q01": np.percentile(states, 1, axis=0).tolist(),
                "q99": np.percentile(states, 99, axis=0).tolist(),
            },
            "actions": {
                "mean": np.mean(actions, axis=0).tolist(),
                "std": np.std(actions, axis=0).tolist(),
                "q01": np.percentile(actions, 1, axis=0).tolist(),
                "q99": np.percentile(actions, 99, axis=0).tolist(),
            }
        },
        "metadata": {
            "source": "dexjoco_mock_data",
            "action_dim": actions.shape[-1],
            "state_dim": states.shape[-1]
        }
    }
    
    # 写入 JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"🎉 归一化统计文件已成功生成至: {output_json_path}")

if __name__ == "__main__":
    # 指向你刚才生成的测试文件
    h5_path = "test_dexjoco.h5" 
    # 将 JSON 保存到项目要求的 norm_stats 目录下
    json_path = "norm_stats/dexjoco_norm.json" 
    compute_norm_stats(h5_path, json_path)