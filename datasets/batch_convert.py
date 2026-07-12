import os
import shutil
from huggingface_hub import snapshot_download
# 导入你写好的转换函数
from convert_dexjoco import convert_dexjoco_to_hdf5 

# === 配置路径 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
REPO_ID = "DexJoCo/DexJoCo-Datasets-Raw"
# Kaggle 上的临时原始数据目录
TEMP_RAW_DIR = "/kaggle/working/temp_raw"
# Kaggle 上的 HDF5 最终输出目录
HDF5_OUT_DIR = "/kaggle/working/hdf5_datasets"

os.makedirs(TEMP_RAW_DIR, exist_ok=True)
os.makedirs(HDF5_OUT_DIR, exist_ok=True)

# 假设你需要转换的 episode 列表（你可以通过 list_repo_files 获取全部列表）
# 这里作为示例，列出两个
episodes_to_convert = [
    "dexjoco_raw_datasets/bimanual_assembly/assembly_demo_10_2026-03-19_15-42-47_880265",
    "dexjoco_raw_datasets/bimanual_assembly/assembly_demo_10_2026-03-19_17-31-23_870703"
]

for ep_path in episodes_to_convert:
    print(f"\n🚀 开始处理: {ep_path}")
    
    # 1. 精准下载单个 Episode
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=TEMP_RAW_DIR,
        allow_patterns=f"{ep_path}/*",
        local_dir_use_symlinks=False
    )
    
    # 拼出本地绝对路径
    local_ep_dir = os.path.join(TEMP_RAW_DIR, ep_path)
    # 取最后一段作为文件名
    ep_name = ep_path.split("/")[-1]
    out_h5_path = os.path.join(HDF5_OUT_DIR, f"{ep_name}.h5")
    
    # 2. 调用你的转换脚本
    convert_dexjoco_to_hdf5(local_ep_dir, out_h5_path)
    
    # 3. 转换完成，立刻删除刚刚下载的原始数据释放空间！
    shutil.rmtree(local_ep_dir)
    print(f"🗑️ 已清理原始文件，释放空间。")

print(f"\n🎉 所有数据转换完毕！HDF5 文件保存在: {HDF5_OUT_DIR}")