import h5py
import numpy as np
import json
import os
import glob

def compute_global_norm_stats(hdf5_dir, output_json_path):
    print(f"⏳ 正在扫描目录: {hdf5_dir}")
    
    # 查找目录下所有的 .h5 文件
    h5_files = glob.glob(os.path.join(hdf5_dir, "*.h5"))
    if not h5_files:
        raise ValueError(f"在 {hdf5_dir} 找不到任何 HDF5 文件！")
        
    all_states = []
    all_actions = []
    
    for f_path in h5_files:
        with h5py.File(f_path, 'r') as f:
            all_states.append(f['data/obs/robot0_proprio'][:])
            all_actions.append(f['data/actions'][:])
            
    # 沿着时间步维度(axis=0)拼接所有数据
    states = np.concatenate(all_states, axis=0)
    actions = np.concatenate(all_actions, axis=0)
        
    print(f"📊 汇总完成! 总状态维度: {states.shape}, 总动作维度: {actions.shape}")
    
    # 计算均值、标准差、分位数
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
            "source": "dexjoco_full_data",
            "action_dim": actions.shape[-1],
            "state_dim": states.shape[-1]
        }
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"🎉 全局归一化统计文件已成功生成至: {output_json_path}")

if __name__ == "__main__":
    # 指向你批量转换后生成的 HDF5 文件夹
    hdf5_directory = "/kaggle/working/hdf5_datasets" 
    json_path = "norm_stats/dexjoco_norm.json" 
    
    compute_global_norm_stats(hdf5_directory, json_path)