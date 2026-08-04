# -*- coding: utf-8 -*-
# FastAPI依赖注入 - 认证与授权
from fastapi import Depends, HTTPException, Request, status

from medical_agent.core.security import decode_token
from medical_agent.infra.mysql import get_db

__all__ = ["get_db", "get_current_user", "get_current_patient", "get_current_doctor", "get_current_admin"]


async def get_current_user(request: Request) -> dict:
    """从Authorization Bearer头中提取JWT并解码，返回用户信息字典"""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少有效的认证令牌",
        )
    token = auth_header[7:]
    try:
        payload = decode_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效或已过期",
        )
    return {"user_id": payload["sub"], "role": payload["role"]}


async def get_current_patient(user: dict = Depends(get_current_user)) -> dict:
    """验证当前用户为患者角色"""
    if user.get("role") != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅患者可访问此资源",
        )
    return user


async def get_current_doctor(user: dict = Depends(get_current_user)) -> dict:
    """验证当前用户为医生角色"""
    if user.get("role") != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅医生可访问此资源",
        )
    return user


async def get_current_admin(user: dict = Depends(get_current_user)) -> dict:
    """验证当前用户为管理员角色"""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="仅管理员可访问此资源",
        )
    return user
