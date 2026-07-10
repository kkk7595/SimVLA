import os
from minio import Minio
from minio.error import S3Error

def upload_folder_to_minio(
    local_folder: str,
    bucket_name: str,
    minio_client,
    minio_prefix: str = "/franka_hand"
):
    """
    递归上传本地文件夹下所有文件到 MinIO
    :param local_folder: 本地文件夹路径（如 franka_hand）
    :param bucket_name: MinIO 存储桶名称
    :param minio_client: MinIO 客户端实例
    :param minio_prefix: MinIO 里的前缀路径
    """
    if not os.path.exists(local_folder):
        print(f"❌ 本地文件夹不存在：{local_folder}")
        return

    # 遍历文件夹所有文件
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            
            # 构造 MinIO 中的文件路径（保持目录结构）
            relative_path = os.path.relpath(local_file_path, local_folder)
            minio_file_path = os.path.join(minio_prefix, relative_path).replace("\\", "/")

            try:
                # 上传文件
                minio_client.fput_object(
                    bucket_name=bucket_name,
                    object_name=minio_file_path,
                    file_path=local_file_path
                )
                print(f"✅ 上传成功：{local_file_path} -> {bucket_name}/{minio_file_path}")

            except S3Error as e:
                print(f"❌ 上传失败：{local_file_path}，错误：{str(e)}")

if __name__ == "__main__":

    # ===================== 配置信息 =====================
    # MinIO 连接配置（外网/内网任选一个）
    # MINIO_ENDPOINT = "123.58.107.62:30090"   # 外网 API 端口
    MINIO_ENDPOINT = "172.16.29.17:30090"   # 北京内网 API 端口 ✅ 修正

    MINIO_ACCESS_KEY = "aubominioadmin"
    MINIO_SECRET_KEY = "WKpc50UC1QPQfWhFQUCW"

    BUCKET_NAME = "vla-dataests"
    MINIO_PREFIX =  "franka_L6_20260503"
    LOCAL_FOLDER = "/home/franka/hkx/Data"
    # ====================================================


    # 初始化 MinIO 客户端
    client = Minio(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False  # 非 https，必须设为 False
    )

    # # 检查存储桶是否存在，不存在则创建
    # try:
    #     if not client.bucket_exists(BUCKET_NAME):
    #         client.make_bucket(BUCKET_NAME)
    #         print(f"📦 存储桶 {BUCKET_NAME} 已创建")
    #     else:
    #         print(f"📦 存储桶 {BUCKET_NAME} 已存在")
    # except S3Error as e:
    #     print(f"存储桶操作失败：{str(e)}")
    #     exit(1)

    # 开始上传
    print(f"\n🚀 开始上传文件夹：{LOCAL_FOLDER}")
    upload_folder_to_minio(
        local_folder=LOCAL_FOLDER,
        bucket_name=BUCKET_NAME,
        minio_client=client,
        minio_prefix=MINIO_PREFIX
    )
    print("\n🎉 文件夹全部上传完成！")