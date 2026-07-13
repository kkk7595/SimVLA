import os
import shutil
import subprocess
from huggingface_hub import HfApi, login, snapshot_download
from huggingface_hub.utils import disable_progress_bars  # 👈 新增：用于关闭长进度条
from convert_dexjoco import convert_dexjoco_to_hdf5

# === 配置 ===
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
REPO_ID = "DexJoCo/DexJoCo-Datasets-Raw"
TEMP_RAW_DIR = "/kaggle/working/temp_raw"
HDF5_OUT_DIR = "/kaggle/working/hdf5_datasets"
HF_REPO_ID = "Kanglr/kkk_data"

# === 关键修正：从环境变量读取 Token ===
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("⚠️ 错误：未找到环境变量 HF_TOKEN，请确保已在 Secrets 中设置！")

# === 分批配置 ===
BATCH_SIZE = 50
BATCH_ID = 10  # ⚠️ 每跑完一批，手动+1并重新运行
START_IDX = 200    # 👈 新增：明确指定从第 200 个文件开始处理（因为之前已经跑了 0~199）

# === 初始化 ===
login(token=hf_token)
api = HfApi()
disable_progress_bars()  # 👈 关闭 Hugging Face 的刷屏日志
os.makedirs(TEMP_RAW_DIR, exist_ok=True)
os.makedirs(HDF5_OUT_DIR, exist_ok=True)

# 1. 获取列表
print("🔍 扫描数据列表...")
all_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
episodes_set = sorted(list(set([f.split("/replay.zarr")[0] for f in all_files if "replay.zarr" in f])))

# 2. 分片
start_idx = (BATCH_ID-10) * BATCH_SIZE + 200
end_idx = min(start_idx + BATCH_SIZE, len(episodes_set))
episodes_to_convert = episodes_set[start_idx:end_idx]
total_eps = len(episodes_to_convert)

print(f"\n📦 开始处理第 {BATCH_ID} 批: 索引 {start_idx} 到 {end_idx} (共 {total_eps} 个文件)\n" + "-"*50)

# 3. 循环转换
# 👈 新增：使用 enumerate 获取当前是第几个文件
for idx, ep_path in enumerate(episodes_to_convert, start=1):
    ep_name = ep_path.split("/")[-1]
    out_h5_path = os.path.join(HDF5_OUT_DIR, f"{ep_name}.h5")
    
    # 打印前缀：[当前/总数]
    print(f"\n▶ [{idx}/{total_eps}] 正在处理: {ep_name}")
    
    if os.path.exists(out_h5_path): 
        print("   ↳ ⚡ 已存在，跳过")
        continue
    
    try:
        snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=TEMP_RAW_DIR, allow_patterns=f"{ep_path}/*")
        convert_dexjoco_to_hdf5(os.path.join(TEMP_RAW_DIR, ep_path), out_h5_path)
    finally:
        if os.path.exists(TEMP_RAW_DIR): shutil.rmtree(TEMP_RAW_DIR)

print("\n" + "-"*50)

# 4. 上传逻辑
zip_path = f"/kaggle/working/batch_{BATCH_ID}.zip"
print(f"📦 压缩数据中...")
subprocess.run(["zip", "-q", "-r", zip_path, HDF5_OUT_DIR], check=True) # 👈 加了 -q 参数，压缩过程也静音不刷屏

print(f"🚀 上传到 Hugging Face: {HF_REPO_ID}...")
api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
api.upload_file(
    path_or_fileobj=zip_path,
    path_in_repo=f"batch_{BATCH_ID}.zip",
    repo_id=HF_REPO_ID,
    repo_type="dataset"
)

# 5. 清理磁盘
shutil.rmtree(HDF5_OUT_DIR)
os.makedirs(HDF5_OUT_DIR)
if os.path.exists(zip_path):
    os.remove(zip_path)

print("✅ 本批次上传完成，磁盘已清理。")