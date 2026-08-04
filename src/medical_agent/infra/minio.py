# -*- coding: utf-8 -*-
# MinIO对象存储客户端 - 文件上传/下载/删除
from io import BytesIO
from typing import Optional

from minio import Minio
from minio.error import S3Error

from medical_agent.core.config import get_settings

_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    """获取MinIO客户端单例"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
    return _client


def ensure_bucket_exists() -> None:
    """确保默认桶存在，不存在则创建"""
    settings = get_settings()
    client = get_minio_client()
    found = client.bucket_exists(settings.MINIO_BUCKET)
    if not found:
        client.make_bucket(settings.MINIO_BUCKET)


def upload_file(object_name: str, data: bytes, content_type: str) -> str:
    """上传文件到MinIO，返回对象名称"""
    settings = get_settings()
    client = get_minio_client()
    ensure_bucket_exists()
    client.put_object(
        settings.MINIO_BUCKET,
        object_name,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_name


def download_file(object_name: str) -> bytes:
    """从MinIO下载文件内容"""
    settings = get_settings()
    client = get_minio_client()
    try:
        response = client.get_object(settings.MINIO_BUCKET, object_name)
        return response.read()
    finally:
        if "response" in locals():
            response.close()
            response.release_conn()


def delete_file(object_name: str) -> None:
    """从MinIO删除文件"""
    settings = get_settings()
    client = get_minio_client()
    client.remove_object(settings.MINIO_BUCKET, object_name)
