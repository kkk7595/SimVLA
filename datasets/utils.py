from __future__ import annotations
import io, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict
import torch

def read_bytes(path: str) -> bytes:
    """
    从指定路径读取字节数据
    使用mmengine的fileio模块，支持多种文件系统（本地/网络/云存储）
    
    参数:
        path: 文件路径，可以是本地路径、URL等
    
    返回:
        bytes: 读取到的字节数据
    """
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    """
    打开HDF5文件，支持直接读取或从字节流读取
    
    参数:
        path: HDF5文件路径
    
    返回:
        h5py.File: 打开的HDF5文件对象（只读模式）
    """
    try:
        # 尝试直接打开文件
        return h5py.File(path, "r")
    except OSError:
        # 如果直接打开失败，先读取字节再从内存中打开
        return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    """
    从指定路径读取视频文件，并将其解码为RGB帧序列
    
    参数:
        path: 视频文件路径
    
    返回:
        np.ndarray: 形状为 [帧数, 高度, 宽度, 3] 的RGB帧数组
    """
    # 读取视频字节数据并创建内存缓冲区
    buf = io.BytesIO(read_bytes(path))
    # 打开视频容器，设置2个线程解码
    container = av.open(buf, options={'threads': '2'})
    
    frames = []
    # 解复用视频流（只处理第一个视频流）
    for packet in container.demux(video=0):
        # 解码每个数据包中的帧并转换为RGB格式的numpy数组
        for f in packet.decode():
            frames.append(f.to_ndarray(format="rgb24"))
    
    # 将帧列表堆叠为numpy数组，维度为 [帧数, H, W, 3]
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    """
    读取Parquet文件并转换为Python字典格式
    
    参数:
        path: Parquet文件路径
    
    返回:
        dict: 以列名为键，列数据为值的字典
    """
    # 读取Parquet文件字节数据
    buf = io.BytesIO(read_bytes(path))
    # 读取Parquet表并转换为Python字典
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    """
    从字节数据解码图像，处理常见的图像尺寸兼容问题
    
    参数:
        x: 图像字节数据（bytes/bytearray）或numpy数组
    
    返回:
        Image.Image: PIL图像对象（RGB格式）
    """
    # 如果输入是字节数据，转换为uint8类型的numpy数组
    if isinstance(x, (bytes, bytearray)):
        x = np.frombuffer(x, dtype=np.uint8)
    
    # 使用OpenCV解码图像为RGB格式
    rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    
    # 如果解码失败，尝试根据字节大小手动重塑数组
    if rgb is None:
        rgb = np.frombuffer(x, dtype=np.uint8)
        # 720x1280x3 = 2764800 像素
        if rgb.size == 2764800:
            rgb = rgb.reshape(720, 1280, 3)
        # 480x640x3 = 921600 像素
        elif rgb.size == 921600:
            rgb = rgb.reshape(480, 640, 3)
    
    # 转换为PIL图像对象返回
    return Image.fromarray(rgb)

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    """
    将四元数转换为6维旋转表示（旋转矩阵的前两列）
    
    参数:
        q: 四元数数组，形状为 [..., 4]
        scalar_first: 四元数是否是标量在前（w, x, y, z），默认为False（x, y, z, w）
    
    返回:
        np.ndarray: 6维旋转表示，形状为 [..., 6]
    """
    # 从四元数创建旋转对象，转换为3x3旋转矩阵
    # 取旋转矩阵的前两列并展平为6维向量
    return R.from_quat(q, scalar_first=scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    """
    将欧拉角转换为6维旋转表示（旋转矩阵的前两列）
    
    参数:
        q: 欧拉角数组，形状为 [..., 3]
        pattern: 欧拉角旋转顺序，默认为"xyz"
    
    返回:
        np.ndarray: 6维旋转表示，形状为 [..., 6]
    """
    # 从欧拉角创建旋转对象（弧度制），转换为3x3旋转矩阵
    # 取旋转矩阵的前两列并展平为6维向量
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    """
    将6维旋转表示转换为XYZ欧拉角（弧度制）
    
    参数:
        v6: 6维旋转表示数组，形状为 [..., 6]
    
    返回:
        np.ndarray: XYZ欧拉角数组，形状为 [..., 3]
    """
    v6 = np.asarray(v6)
    # 检查输入维度是否正确
    if v6.shape[-1] != 6:
        raise ValueError("最后一维必须是6（输入为 %s）" % (v6.shape[-1],))
    
    # 提取6维向量中的前3个元素（第一列）和后3个元素（第二列）
    a1 = v6[..., 0:5:2]  # 索引 0,2,4
    a2 = v6[..., 1:6:2]  # 索引 1,3,5
    
    # 归一化第一列得到旋转矩阵的第一行
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    # 计算第二列在第一列上的投影
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    # 正交化得到旋转矩阵的第二行
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    # 叉乘得到旋转矩阵的第三行（保证右手坐标系）
    b3 = np.cross(b1, b2)
    
    # 堆叠得到完整的3x3旋转矩阵，形状为 (..., 3, 3)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    # 将旋转矩阵转换为XYZ欧拉角（弧度制）
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    """
    将6维旋转表示转换为四元数
    
    参数:
        v6: 6维旋转表示数组，形状为 [..., 6]
        scalar_first: 输出四元数是否标量在前（w, x, y, z），默认为False（x, y, z, w）
    
    返回:
        np.ndarray: 四元数数组，形状为 [..., 4]
    """
    v6 = np.asarray(v6)
    # 检查输入维度是否正确
    if v6.shape[-1] != 6:
        raise ValueError("最后一维必须是6（输入为 %s）" % (v6.shape[-1],))
    
    # 提取6维向量中的前3个元素（第一列）和后3个元素（第二列）
    a1 = v6[..., 0:5:2]  # 索引 0,2,4
    a2 = v6[..., 1:6:2]  # 索引 1,3,5
    
    # 归一化第一列得到旋转矩阵的第一行
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    # 计算第二列在第一列上的投影
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    # 正交化得到旋转矩阵的第二行
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    # 叉乘得到旋转矩阵的第三行（保证右手坐标系）
    b3 = np.cross(b1, b2)
    
    # 堆叠得到完整的3x3旋转矩阵，形状为 (..., 3, 3)
    rot_mats = np.stack((b1, b2, b3), axis=-1)
    # 将旋转矩阵转换为四元数
    return R.from_matrix(rot_mats).as_quat(scalar_first=scalar_first)

def action_slice(abs_traj: torch.Tensor, idx_for_delta: Sequence[int] = ()) -> Dict[str, torch.Tensor]:
    """
    从绝对轨迹中提取本体感觉信息和动作指令，支持对指定维度计算相对值
    
    参数:
        abs_traj: 绝对轨迹张量，形状必须为 [H+1, D]（H>=1），H为步数，D为特征维度
        idx_for_delta: 需要计算相对值的维度索引列表
    
    返回:
        Dict[str, torch.Tensor]: 
            - proprio: 初始状态的本体感觉信息，形状 [D]
            - action: 动作指令序列，形状 [H, D]
    """
    # 类型检查：确保输入是torch张量
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj必须是torch.Tensor类型")
    
    # 维度检查：必须是2维且第一维长度至少为2
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj必须是[H+1, D]形状且H>=1")
    
    # 提取初始状态作为本体感觉信息（第一行）
    proprio = abs_traj[0]         # 形状 [D]
    # 提取后续状态作为动作指令（从第二行开始），使用clone避免原地操作
    action = abs_traj[1:].clone() # 形状 [H, D]
    
    # 如果指定了需要计算相对值的维度
    if idx_for_delta:
        # 将维度索引转换为张量，保持设备一致性
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        # 计算指定维度的相对值（当前值 - 初始值）
        action[:, idx] -= proprio[idx]
    
    # 返回本体感觉信息和动作指令
    return {"proprio": proprio, "action": action}





