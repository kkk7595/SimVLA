#kkk
import h5py
import numpy as np
import os

def axis_angle_to_rot_vec(axis, angle):
    """
    将 3维轴 和 1维角度 转换为 3维旋转向量
    """
    axis = np.array(axis, dtype=np.float32)
    # 确保轴是单位向量
    norm = np.linalg.norm(axis)
    if norm > 1e-6:
        axis = axis / norm
    return axis * angle

def convert_dexjoco_to_hdf5(dummy_output_path="test_dexjoco.h5"):
    """
    模拟 DexJoco 数据向 HDF5 的转换。
    一旦学长给了真实数据，把这里的 dummy_data 替换为真实读取逻辑即可。
    """
    print("⏳ 开始生成并转换数据...")
    
    # 1. 假设 DexJoco 采集了 100 帧轨迹数据
    num_frames = 100
    
    # 模拟环境观测 (假设 384x384 图像)
    dummy_obs_image = np.random.randint(0, 255, (num_frames, 384, 384, 3), dtype=np.uint8)
    dummy_proprio = np.random.rand(num_frames, 7).astype(np.float32) # 模拟7维本体感受
    
    # 2. 模拟原始 DexJoco 动作 (包含轴角)
    # 假设原本动作是 8 维： [x, y, z (3维)] + [axis_x, axis_y, axis_z (3维)] + [angle (1维)] + [gripper (1维)]
    dummy_raw_actions = np.random.rand(num_frames, 8).astype(np.float32)
    
    # 3. 核心转换逻辑：轴角 -> 旋转向量
    converted_actions = []
    for i in range(num_frames):
        pos = dummy_raw_actions[i, 0:3]           # 位置
        axis = dummy_raw_actions[i, 3:6]          # 旋转轴
        angle = dummy_raw_actions[i, 6]           # 旋转角
        gripper = dummy_raw_actions[i, 7:8]       # 夹爪
        
        # 转换为旋转向量
        rot_vec = axis_angle_to_rot_vec(axis, angle)
        
        # 拼接新的动作：[x, y, z, rot_x, rot_y, rot_z, gripper] -> 共 7 维
        new_action = np.concatenate([pos, rot_vec, gripper])
        converted_actions.append(new_action)
        
    converted_actions = np.array(converted_actions, dtype=np.float32)
    print(f"✅ 动作维度转换成功: 原始 {dummy_raw_actions.shape[-1]}维 -> 新格式 {converted_actions.shape[-1]}维")

    # 4. 写入 HDF5 格式 (匹配 SimVLA 数据加载器的预期)
    with h5py.File(dummy_output_path, 'w') as f:
        # 参考 libero_hdf5.py 的常见层级结构
        data_group = f.create_group('data')
        
        # 写入观测
        obs_group = data_group.create_group('obs')
        obs_group.create_dataset('agentview_rgb', data=dummy_obs_image, compression="gzip")
        obs_group.create_dataset('robot0_proprio', data=dummy_proprio)
        
        # 写入转换后的动作
        data_group.create_dataset('actions', data=converted_actions)
        
    print(f"🎉 HDF5 文件已成功保存至: {dummy_output_path}")

if __name__ == "__main__":
    convert_dexjoco_to_hdf5()