import os
import h5py
import numpy as np
import zarr
import cv2

def convert_dexjoco_to_hdf5(episode_dir, output_hdf5_path):
    print(f"⏳ 开始转换数据: {episode_dir}...")
    
    zarr_path = os.path.join(episode_dir, "replay.zarr")
    video_dir = os.path.join(episode_dir, "videos")
    
    # ==================== 1. 读取 Zarr 数据 ====================
    print("  -> 读取 Zarr 状态和动作...")
    z_root = zarr.open(zarr_path, mode='r')
    actions = z_root['data']['action_rotvec'][:]
    proprio = z_root['data']['state'][:].squeeze(1)
    num_frames = actions.shape[0]
    
    # ==================== 2. 动态搜索并读取 视频图像 ====================
    # 优先找 ego.mp4，找不到就随便挑一个 .mp4，如果连 mp4 都没有就抛出明确错误停止程序
    video_path = os.path.join(video_dir, "ego.mp4")
    if not os.path.exists(video_path):
        if not os.path.exists(video_dir):
            raise FileNotFoundError(f"❌ 严重错误: 视频文件夹不存在 {video_dir}")
        
        mp4_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        if not mp4_files:
            raise FileNotFoundError(f"❌ 严重错误: 在 {video_dir} 找不到任何 .mp4 视频文件")
        
        video_path = os.path.join(video_dir, mp4_files[0])
        print(f"  -> ⚠️ 未找到 ego.mp4，自动切换使用: {mp4_files[0]}")

    print(f"  -> 读取视频图像 {os.path.basename(video_path)} (目标帧数: {num_frames})...")
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    
    frames = np.array(frames, dtype=np.uint8)
    
    # ==================== 3. 校验并对齐数据 ====================
    if frames.shape[0] == 0:
        raise ValueError(f"❌ 严重错误: 成功找到视频文件 {video_path}，但读取出的帧数为 0！视频可能已损坏。")

    if frames.shape[0] != num_frames:
        print(f"  -> ⚠️ 警告: 视频帧数 ({frames.shape[0]}) 与动作序列 ({num_frames}) 不一致！将进行截断对齐。")
        min_len = min(frames.shape[0], num_frames)
        frames = frames[:min_len]
        actions = actions[:min_len]
        proprio = proprio[:min_len]
        
    print(f"✅ 数据读取对齐完成:\n    - 图像: {frames.shape}\n    - 状态: {proprio.shape}\n    - 动作: {actions.shape}")

    # ==================== 4. 写入 HDF5 格式 ====================
    print("  -> 写入 HDF5 文件...")
    with h5py.File(output_hdf5_path, 'w') as f:
        data_group = f.create_group('data')
        obs_group = data_group.create_group('obs')
        obs_group.create_dataset('agentview_rgb', data=frames, compression="gzip")
        obs_group.create_dataset('robot0_proprio', data=proprio)
        data_group.create_dataset('actions', data=actions)
        
    print(f"🎉 HDF5 文件已成功保存至: {output_hdf5_path}\n")