# -*- coding: utf-8 -*-
# 认证路由 - 用户注册、登录、令牌刷新与个人信息
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.api.deps import get_current_user, get_db
from medical_agent.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from medical_agent.models.user import User

router = APIRouter(prefix="/api/v1/auth", tags=["认证"])


class RegisterRequest(BaseModel):
    """注册请求"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    role: str = Field(default="patient", description="用户角色")
    real_name: str = Field(default="", max_length=50, description="真实姓名")


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")


class RefreshRequest(BaseModel):
    """刷新令牌请求"""
    refresh_token: str = Field(..., description="刷新令牌")


def _build_user_response(user: User) -> dict:
    """构建用户信息响应字典"""
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "real_name": user.real_name,
    }


@router.post("/register", summary="用户注册")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """注册新用户"""
    # 检查用户名唯一性
    stmt = select(User).where(User.username == req.username, User.is_deleted == False)
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="用户名已存在",
        )

    # 创建用户
    user = User(
        username=req.username,
        password_hash=hash_password(req.password),
        role=req.role,
        real_name=req.real_name or req.username,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    # 生成令牌
    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id, user.role)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": _build_user_response(user),
    }


@router.post("/login", summary="用户登录")
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    """用户登录"""
    stmt = select(User).where(User.username == req.username, User.is_deleted == False)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="账号已被禁用",
        )

    access_token = create_access_token(user.id, user.role)
    refresh_token = create_refresh_token(user.id, user.role)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "user": _build_user_response(user),
    }


@router.post("/refresh", summary="刷新令牌")
async def refresh(req: RefreshRequest):
    """使用刷新令牌获取新的访问令牌"""
    try:
        payload = decode_token(req.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌无效或已过期",
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌类型不正确，需要refresh令牌",
        )

    user_id = payload["sub"]
    role = payload["role"]
    access_token = create_access_token(user_id, role)

    return {
        "access_token": access_token,
    }


@router.get("/me", summary="获取当前用户信息")
async def me(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """获取当前登录用户的详细信息"""
    stmt = select(User).where(User.id == current_user["user_id"], User.is_deleted == False)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )

    return {
        "user": _build_user_response(user),
    }
