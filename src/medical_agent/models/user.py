# -*- coding: utf-8 -*-
# 用户模型 - 系统用户认证与权限管理
from typing import Optional

from sqlalchemy import Boolean, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from medical_agent.core.base_model import BaseModel


class User(BaseModel):
    """系统用户模型"""

    __tablename__ = "users"

    # 用户名，唯一
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, comment="用户名")
    # 密码哈希值
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False, comment="密码哈希")
    # 用户角色，默认patient
    role: Mapped[str] = mapped_column(String(20), nullable=False, default="patient", comment="用户角色")
    # 真实姓名
    real_name: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="真实姓名")
    # 联系电话
    phone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, comment="联系电话")
    # 邮箱
    email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, comment="邮箱")
    # 是否激活
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, comment="是否激活")
    # 是否已删除（软删除）
    is_deleted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, comment="是否已删除")

    __table_args__ = (
        Index("idx_users_username", "username"),
        Index("idx_users_role", "role"),
    )
