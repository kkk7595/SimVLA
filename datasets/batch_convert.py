import os
import shutil
from huggingface_hub import snapshot_download, HfApi
from convert_dexjoco import convert_dexjoco_to_hdf5 

# === 配置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
REPO_ID = "DexJoCo/DexJoCo-Datasets-Raw"
TEMP_RAW_DIR = "/kaggle/working/temp_raw"
HDF5_OUT_DIR = "/kaggle/working/hdf5_datasets"

# === 分批配置 (关键：解决空间溢出) ===
BATCH_SIZE = 50   # 每次处理 50 个文件，如果还报错，请改为 20
BATCH_ID = 0      # ⚠️ 每次跑完后，手动把这里加 1，再重新提交运行！

os.makedirs(TEMP_RAW_DIR, exist_ok=True)
os.makedirs(HDF5_OUT_DIR, exist_ok=True)

print("🔍 正在连接 Hugging Face 获取完整数据列表...")
api = HfApi(endpoint="https://hf-mirror.com")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")

episodes_set = sorted(list(set([f.split("/replay.zarr")[0] for f in all_files if "replay.zarr" in f])))

# === 分片逻辑 ===
start_idx = BATCH_ID * BATCH_SIZE
end_idx = start_idx + BATCH_SIZE
episodes_to_convert = episodes_set[start_idx:end_idx]

print(f"📦 共 {len(episodes_set)} 个文件，本次任务：处理第 {BATCH_ID} 批 (索引 {start_idx} 到 {end_idx})")

for idx, ep_path in enumerate(episodes_to_convert, 1):
    print(f"\n🚀 [{idx}/{len(episodes_to_convert)}] 开始处理: {ep_path}")
    
    ep_name = ep_path.split("/")[-1]
    out_h5_path = os.path.join(HDF5_OUT_DIR, f"{ep_name}.h5")
    
    if os.path.exists(out_h5_path):
        print(f"⏭️ {ep_name}.h5 已存在，跳过。")
        continue

    # 每次下载前确保清空临时目录
    if os.path.exists(TEMP_RAW_DIR):
        shutil.rmtree(TEMP_RAW_DIR)
    os.makedirs(TEMP_RAW_DIR, exist_ok=True)

    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=TEMP_RAW_DIR, 
                          allow_patterns=f"{ep_path}/*", local_dir_use_symlinks=False)
        convert_dexjoco_to_hdf5(os.path.join(TEMP_RAW_DIR, ep_path), out_h5_path)
    except Exception as e:
        print(f"❌ 错误: {e}")
    finally:
        if os.path.exists(TEMP_RAW_DIR):
            shutil.rmtree(TEMP_RAW_DIR)