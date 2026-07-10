from minio import Minio
from minio.error import S3Error
import os

def download_from_minio():
    # ===================== 你的 MinIO 配置 =====================
    MINIO_ENDPOINT = "172.16.29.17:30090"
    MINIO_ACCESS_KEY = "aubominioadmin"
    MINIO_SECRET_KEY = "WKpc50UC1QPQfWhFQUCW"
    MINIO_SECURE = False

    # ===================== 要下载的文件信息 =====================
    BUCKET_NAME = "vla-dataests"
    PREFIX = "franka_L6_20260503/libero/"  # 文件夹路径，末尾必须加 /

    # ===================== 本地保存路径 =====================
    LOCAL_BASE_PATH = "/home/keep/Desktop/project/X-RLinf/dataset/franka-l6"

    try:
        # 1. 连接 MinIO
        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE
        )

        print(f"正在下载文件夹：{BUCKET_NAME}/{PREFIX}")

        # 2. 列出所有文件
        objects = client.list_objects(BUCKET_NAME, prefix=PREFIX, recursive=True)

        # 3. 逐个下载
        for obj in objects:
            local_file = os.path.join(LOCAL_BASE_PATH, obj.object_name)
            local_dir = os.path.dirname(local_file)

            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            print(f"下载：{obj.object_name}")
            client.fget_object(BUCKET_NAME, obj.object_name, local_file)

        print("✅ 整个文件夹下载完成！")

    except S3Error as e:
        print("❌ MinIO 错误：", e)
    except Exception as e:
        print("❌ 其他错误：", e)

if __name__ == "__main__":
    download_from_minio()