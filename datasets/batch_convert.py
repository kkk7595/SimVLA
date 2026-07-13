import os
import shutil
from huggingface_hub import HfApi, login, snapshot_download
from convert_dexjoco import convert_dexjoco_to_hdf5

# === 配置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
REPO_ID = "DexJoCo/DexJoCo-Datasets-Raw"
TEMP_RAW_DIR = "/kaggle/working/temp_raw"
HDF5_OUT_DIR = "/kaggle/working/hdf5_datasets"
HF_REPO_ID = "Kanglr/kkk_data"

# === 关键修正：从环境变量读取 Token ===
# 在 Kaggle 中通过 Secrets 设置 HF_TOKEN，或在运行前执行 export HF_TOKEN=...
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ 错误：未找到环境变量 HF_TOKEN，请确保已在 Secrets 中设置！")

# === 分批配置 ===
BATCH_SIZE = 20
BATCH_ID = 0  # ⚠️ 每跑完一批，手动+1并重新运行

# === 初始化 ===
login(token=hf_token)
api = HfApi()
os.makedirs(TEMP_RAW_DIR, exist_ok=True)
os.makedirs(HDF5_OUT_DIR, exist_ok=True)

# 1. 获取列表
print("🔍 扫描数据列表...")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
episodes_set = sorted(list(set([f.split("/replay.zarr")[0] for f in all_files if "replay.zarr" in f])))

# 2. 分片
start_idx = BATCH_ID * BATCH_SIZE
end_idx = min(start_idx + BATCH_SIZE, len(episodes_set))
episodes_to_convert = episodes_set[start_idx:end_idx]

print(f"📦 处理第 {BATCH_ID} 批: {start_idx} 到 {end_idx}")

# 3. 循环转换
for ep_path in episodes_to_convert:
    ep_name = ep_path.split("/")[-1]
    out_h5_path = os.path.join(HDF5_OUT_DIR, f"{ep_name}.h5")
    
    if os.path.exists(out_h5_path): continue
    
    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=TEMP_RAW_DIR, allow_patterns=f"{ep_path}/*")
        convert_dexjoco_to_hdf5(os.path.join(TEMP_RAW_DIR, ep_path), out_h5_path)
    finally:
        if os.path.exists(TEMP_RAW_DIR): shutil.rmtree(TEMP_RAW_DIR)

# 4. 上传逻辑
zip_path = f"/kaggle/working/batch_{BATCH_ID}.zip"
print(f"📦 压缩中...")
# 👈 使用 subprocess 替代 !zip
subprocess.run(["zip", "-r", zip_path, HDF5_OUT_DIR], check=True)

print(f"🚀 上传到 Hugging Face: {HF_REPO_ID}...")
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=zip_path,
    path_in_repo=f"batch_{BATCH_ID}.zip",
    repo_id=HF_REPO_ID,
    repo_type="dataset"
)

# 5. 清理磁盘
# 👈 使用 Python 原生函数替代 !rm
shutil.rmtree(HDF5_OUT_DIR)
os.makedirs(HDF5_OUT_DIR) # 重新创建空文件夹
if os.path.exists(zip_path):
    os.remove(zip_path)

print("✅ 本批次上传完成，磁盘已清理。")