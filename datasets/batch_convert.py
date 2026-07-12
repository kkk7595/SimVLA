import os
import shutil
from huggingface_hub import snapshot_download, HfApi
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

# ================= 自动获取所有 Episode 路径 =================
print("🔍 正在连接 Hugging Face 获取完整数据列表...")
api = HfApi(endpoint="https://hf-mirror.com")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

# 筛选出所有包含 replay.zarr 的文件夹路径
episodes_set = set()
for f in all_files:
    if "replay.zarr" in f:
        # 截取 replay.zarr 之前的部分作为 Episode 路径
        ep_path = f.split("/replay.zarr")[0]
        episodes_set.add(ep_path)

episodes_to_convert = sorted(list(episodes_set))
print(f"📦 扫描完毕！共发现 {len(episodes_to_convert)} 个 Episode 需要转换！")
# =============================================================

for idx, ep_path in enumerate(episodes_to_convert, 1):
    print(f"\n🚀 [{idx}/{len(episodes_to_convert)}] 开始处理: {ep_path}")
    
    ep_name = ep_path.split("/")[-1]
    out_h5_path = os.path.join(HDF5_OUT_DIR, f"{ep_name}.h5")
    
    # 💡 如果这个 h5 文件已经存在，说明之前转过了，直接跳过（支持断点续传）
    if os.path.exists(out_h5_path):
        print(f"⏭️ {ep_name}.h5 已存在，跳过转换。")
        continue

    # 1. 精准下载单个 Episode
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=TEMP_RAW_DIR,
        allow_patterns=f"{ep_path}/*",
        local_dir_use_symlinks=False
    )
    
    local_ep_dir = os.path.join(TEMP_RAW_DIR, ep_path)
    
    # 2. 调用你的转换脚本
    try:
        convert_dexjoco_to_hdf5(local_ep_dir, out_h5_path)
    except Exception as e:
        print(f"❌ 转换 {ep_name} 时发生错误: {e}")
    
    # 3. 转换完成，立刻删除刚刚下载的原始数据释放空间！
    if os.path.exists(local_ep_dir):
        shutil.rmtree(local_ep_dir)
        print(f"🗑️ 已清理原始文件，释放空间。")

print(f"\n🎉 恭喜！所有 {len(episodes_to_convert)} 个数据转换完毕！HDF5 文件保存在: {HDF5_OUT_DIR}")