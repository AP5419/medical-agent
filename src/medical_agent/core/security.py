# -*- coding: utf-8 -*-
# JWT认证与密码哈希模块
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import bcrypt
import jwt

from medical_agent.core.config import get_settings


def hash_password(plain: str) -> str:
    """对明文密码进行bcrypt哈希"""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """验证明文密码是否匹配哈希"""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _get_secret_key() -> str:
    """从配置中派生JWT签名密钥"""
    settings = get_settings()
    base = f"{settings.APP_NAME}_{settings.DASHSCOPE_API_KEY}_{settings.DB_PASSWORD}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def create_access_token(user_id: int, role: str) -> str:
    """创建访问令牌，有效期24小时"""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=24),
        "type": "access",
    }
    secret = _get_secret_key()
    return jwt.encode(payload, secret, algorithm="HS256")


def create_refresh_token(user_id: int, role: str) -> str:
    """创建刷新令牌，有效期7天"""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + timedelta(days=7),
        "type": "refresh",
    }
    secret = _get_secret_key()
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> Dict[str, Any]:
    """解码并验证JWT令牌，返回payload字典"""
    secret = _get_secret_key()
    return jwt.decode(token, secret, algorithms=["HS256"])


def get_token_user_id(token: str) -> int:
    """从令牌中提取用户ID"""
    payload = decode_token(token)
    return int(payload["sub"])


def get_token_role(token: str) -> str:
    """从令牌中提取用户角色"""
    payload = decode_token(token)
    return payload["role"]
