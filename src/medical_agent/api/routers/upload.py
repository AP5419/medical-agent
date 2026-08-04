# -*- coding: utf-8 -*-
# 文件上传路由 - 报告文件上传与下载
import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse

from medical_agent.api.deps import get_current_user
from medical_agent.engines.rag.mineru_client import MinerUClient
from medical_agent.infra.minio import download_file, upload_file

router = APIRouter(prefix="/api/v1/upload", tags=["文件上传"])

# 允许的文件类型
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "application/pdf",
    "application/dicom",
}

# 最大文件大小 20MB
MAX_FILE_SIZE = 20 * 1024 * 1024


def _get_file_extension(content_type: str) -> str:
    """根据content_type获取文件扩展名"""
    extension_map = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "application/pdf": ".pdf",
        "application/dicom": ".dcm",
    }
    return extension_map.get(content_type, ".bin")


@router.post("/report", summary="上传报告文件")
async def upload_report(
    file: UploadFile,
    current_user: dict = Depends(get_current_user),
):
    """上传医疗报告文件到MinIO"""
    # 验证文件类型
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file.content_type}。支持: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # 读取文件内容并验证大小
    file_data = await file.read()
    file_size = len(file_data)

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件大小超过限制 ({file_size / 1024 / 1024:.1f}MB > 20MB)",
        )

    # 生成唯一文件名
    file_ext = _get_file_extension(file.content_type)
    object_name = f"reports/{current_user['user_id']}/{uuid.uuid4().hex}{file_ext}"

    # 上传到MinIO
    upload_file(object_name, file_data, file.content_type)

    return {
        "success": True,
        "file_url": f"/api/v1/upload/report/{object_name}",
        "object_name": object_name,
    }


# 文档上传允许的文件类型
DOCUMENT_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

# 文档最大大小 50MB
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024


def _get_document_extension(content_type: str) -> str:
    """根据content_type获取文档文件扩展名"""
    extension_map = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    }
    return extension_map.get(content_type, ".bin")


@router.post("/document", summary="上传文档并解析")
async def upload_document(
    file: UploadFile,
    current_user: dict = Depends(get_current_user),
):
    """上传医疗文档（PDF/DOCX/PPTX/XLSX），使用MinerU解析为Markdown"""
    # 验证文件类型
    if file.content_type not in DOCUMENT_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型: {file.content_type}。支持: PDF, DOCX, PPTX, XLSX",
        )

    # 读取文件内容并验证大小
    file_data = await file.read()
    file_size = len(file_data)

    if file_size > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"文件大小超过限制 ({file_size / 1024 / 1024:.1f}MB > 50MB)",
        )

    # 生成唯一文件名并保存到临时目录
    file_ext = _get_document_extension(file.content_type)
    object_name = f"documents/{current_user['user_id']}/{uuid.uuid4().hex}{file_ext}"

    # 上传到MinIO
    upload_file(object_name, file_data, file.content_type)

    # 写入临时文件供 MinerU 解析
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
        tmp_file.write(file_data)
        tmp_path = tmp_file.name

    try:
        # 调用 MinerU 解析文档
        client = MinerUClient()
        result = await client.parse_file(tmp_path)

        if result["success"]:
            markdown_text = result["markdown"]
            return {
                "success": True,
                "file_url": f"/api/v1/upload/report/{object_name}",
                "object_name": object_name,
                "markdown_preview": markdown_text[:500],
                "char_count": len(markdown_text),
            }
        else:
            return {
                "success": False,
                "file_url": f"/api/v1/upload/report/{object_name}",
                "object_name": object_name,
                "markdown_preview": "",
                "char_count": 0,
                "error": result["error"],
            }
    finally:
        # 清理临时文件
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.get("/report/{object_name:path}", summary="下载报告文件")
async def download_report(
    object_name: str,
    current_user: dict = Depends(get_current_user),
):
    """从MinIO下载报告文件"""
    try:
        file_data = download_file(object_name)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="文件不存在",
        )

    # 推断content_type
    content_type = "application/octet-stream"
    if object_name.endswith(".jpg") or object_name.endswith(".jpeg"):
        content_type = "image/jpeg"
    elif object_name.endswith(".png"):
        content_type = "image/png"
    elif object_name.endswith(".pdf"):
        content_type = "application/pdf"
    elif object_name.endswith(".dcm"):
        content_type = "application/dicom"

    return StreamingResponse(
        iter([file_data]),
        media_type=content_type,
        headers={"Content-Disposition": f'inline; filename="{object_name.split("/")[-1]}"'},
    )
