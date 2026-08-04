# -*- coding: utf-8 -*-
# 安全模块测试 - 密码哈希 + JWT 令牌
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import jwt
import pytest

from medical_agent.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_role,
    get_token_user_id,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """密码哈希测试"""

    def test_hash_and_verify(self):
        """测试哈希后可以正确验证"""
        plain = "my_secure_password123"
        hashed = hash_password(plain)
        assert hashed != plain
        assert verify_password(plain, hashed) is True

    def test_verify_wrong_password(self):
        """测试错误密码验证失败"""
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_hash_is_unique(self):
        """测试每次哈希结果不同（盐值随机）"""
        h1 = hash_password("same_password")
        h2 = hash_password("same_password")
        assert h1 != h2

    def test_hash_output_is_string(self):
        """测试哈希结果为字符串"""
        hashed = hash_password("test")
        assert isinstance(hashed, str)
        assert len(hashed) > 0


class TestJWTToken:
    """JWT 令牌测试"""

    def test_create_and_decode_access_token(self):
        """测试创建并解码访问令牌"""
        token = create_access_token(user_id=1, role="patient")
        payload = decode_token(token)
        assert payload["sub"] == "1"
        assert payload["role"] == "patient"
        assert payload["type"] == "access"

    def test_create_refresh_token(self):
        """测试刷新令牌类型"""
        token = create_refresh_token(user_id=2, role="doctor")
        payload = decode_token(token)
        assert payload["type"] == "refresh"
        assert payload["sub"] == "2"

    def test_token_expiry(self):
        """测试令牌过期时间"""
        token = create_access_token(user_id=1, role="admin")
        payload = decode_token(token)
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc).replace(tzinfo=None)
        assert exp > now
        # Token 应在 24 小时内过期
        delta_seconds = (exp - now).total_seconds()
        assert 23 * 3600 < delta_seconds < 25 * 3600

    def test_get_token_user_id(self):
        """测试提取用户 ID"""
        token = create_access_token(user_id=42, role="patient")
        assert get_token_user_id(token) == 42

    def test_get_token_role(self):
        """测试提取角色"""
        token = create_access_token(user_id=1, role="doctor")
        assert get_token_role(token) == "doctor"

    def test_decode_invalid_token(self):
        """测试解码无效令牌抛出异常"""
        with pytest.raises(Exception):
            decode_token("invalid.token.here")

    def test_admin_role_token(self):
        """测试管理员角色令牌"""
        token = create_access_token(user_id=99, role="admin")
        assert get_token_role(token) == "admin"
        assert get_token_user_id(token) == 99
