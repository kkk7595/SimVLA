##  删除 MinIO 存储中指定的整个 “文件夹”（包括里面所有的文件、子文件夹），是一个批量删除工具。

from minio import Minio
from minio.error import S3Error

def delete_minio_folder(bucket_name: str, prefix: str, minio_client):
    """
    删除 MinIO 里的整个“文件夹”（前缀下所有文件）
    :param bucket_name: 存储桶名
    :param prefix: 要删除的前缀路径（如 redcube12/）
    :param minio_client: MinIO 客户端
    """
    try:
        # 1. 列出该前缀下所有文件
        objects = minio_client.list_objects(
            bucket_name,
            prefix=prefix,
            recursive=True
        )

        # 2. 批量删除
        for obj in objects:
            minio_client.remove_object(bucket_name, obj.object_name)
            print(f"🗑️ 已删除：{obj.object_name}")

        print(f"\n✅ 文件夹删除完成：{bucket_name}/{prefix}")

    except S3Error as e:
        print(f"❌ 删除失败：{str(e)}")

if __name__ == "__main__":
    # ===================== 你的配置（直接用） =====================
    MINIO_ENDPOINT = "172.16.29.17:30090"  # 必须是 API 端口
    MINIO_ACCESS_KEY = "aubominioadmin"
    MINIO_SECRET_KEY = "WKpc50UC1QPQfWhFQUCW"

    BUCKET_NAME = "vla-dataests"
    DELETE_PREFIX = "franka_hdf5_20260503/"  # 你要删的文件夹

    # ============================================================

    # 初始化客户端
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    # 执行删除
    delete_minio_folder(BUCKET_NAME, DELETE_PREFIX, client)