import os
import shutil
import subprocess  # 👈 新增导入
from huggingface_hub import HfApi, login, snapshot_download
from convert_dexjoco import convert_dexjoco_to_hdf5

# ... (前面的代码保持不变) ...

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