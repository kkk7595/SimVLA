import os
import h5py
import numpy as np
import zarr
import cv2

def convert_dexjoco_to_hdf5(episode_dir, output_hdf5_path):
    print(f"⏳ 开始转换数据: {episode_dir}...")
    
    # 拼出 Zarr 数据和 视频数据 的路径
    zarr_path = os.path.join(episode_dir, "replay.zarr")
    video_path = os.path.join(episode_dir, "videos", "ego.mp4") # 使用全局视角作为示例
    
    # ==================== 1. 读取 Zarr 数据 ====================
    print("  -> 读取 Zarr 状态和动作...")
    z_root = zarr.open(zarr_path, mode='r')
    
    # 💡 直接提取官方提供的 action_rotvec，无需自己写转换函数了[cite: 2]
    actions = z_root['data']['action_rotvec'][:]
    
    # 💡 提取 state，并使用 squeeze() 去掉多余的维度：(708, 1, 61) 变成 (708, 61)
    proprio = z_root['data']['state'][:].squeeze(1)
    
    num_frames = actions.shape[0]
    
    # ==================== 2. 读取 视频图像 ====================
    print(f"  -> 读取视频图像 (目标帧数: {num_frames})...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV 默认读取为 BGR 格式，转换为深度学习常用的 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    frames = np.array(frames, dtype=np.uint8)
    
    # ==================== 3. 校验并对齐数据 ====================
    # 确保视频帧数和动作序列长度完全一致
    if frames.shape[0] != num_frames:
        print(f"⚠️ 警告: 视频帧数 ({frames.shape[0]}) 与动作序列 ({num_frames}) 不一致！将进行截断对齐。")
        min_len = min(frames.shape[0], num_frames)
        frames = frames[:min_len]
        actions = actions[:min_len]
        proprio = proprio[:min_len]
        
    print(f"✅ 数据读取对齐完成:\n    - 图像: {frames.shape}\n    - 状态: {proprio.shape}\n    - 动作: {actions.shape}")

    # ==================== 4. 写入 HDF5 格式 ====================
    print("  -> 写入 HDF5 文件...")
    with h5py.File(output_hdf5_path, 'w') as f:
        # 参考 SimVLA/LIBERO 的常见层级结构[cite: 2]
        data_group = f.create_group('data')
        
        # 写入观测 (图像和本体感受)[cite: 2]
        obs_group = data_group.create_group('obs')
        obs_group.create_dataset('agentview_rgb', data=frames, compression="gzip")
        obs_group.create_dataset('robot0_proprio', data=proprio)
        
        # 写入动作[cite: 2]
        data_group.create_dataset('actions', data=actions)
        
    print(f"🎉 HDF5 文件已成功保存至: {output_hdf5_path}\n")

if __name__ == "__main__":
    # 指向你刚才用 download_sample.py 下载的那个具体 Episode 的文件夹路径
    test_episode_dir = "/home/kkk/simvla/sample_data/dexjoco_raw_datasets/bimanual_assembly/assembly_demo_10_2026-03-19_15-42-47_880265"
    
    # 输出的测试 HDF5 文件名[cite: 2]
    output_file = "test_dexjoco.h5"
    
    # 执行转换[cite: 2]
    convert_dexjoco_to_hdf5(test_episode_dir, output_file)