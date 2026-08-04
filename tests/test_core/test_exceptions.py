# -*- coding: utf-8 -*-
# 异常模块测试
import pytest

from medical_agent.core.exceptions import (
    BadRequestException,
    BusinessException,
    ConflictException,
    ForbiddenException,
    NotFoundException,
    UnauthorizedException,
)


class TestBusinessException:
    """业务异常基类测试"""

    def test_default_status_code(self):
        """测试默认状态码为 400"""
        exc = BusinessException("错误")
        assert exc.status_code == 400
        assert exc.message == "错误"

    def test_custom_status_code(self):
        """测试自定义状态码"""
        exc = BusinessException("自定义错误", status_code=418, code="418")
        assert exc.status_code == 418
        assert exc.code == "418"


class TestNotFoundException:
    """NotFoundException 测试"""

    def test_status_code_404(self):
        """测试状态码为 404"""
        exc = NotFoundException()
        assert exc.status_code == 404

    def test_custom_message(self):
        """测试自定义消息"""
        exc = NotFoundException("用户不存在")
        assert exc.message == "用户不存在"
        assert exc.status_code == 404


class TestUnauthorizedException:
    """UnauthorizedException 测试"""

    def test_status_code_401(self):
        """测试状态码为 401"""
        exc = UnauthorizedException()
        assert exc.status_code == 401


class TestForbiddenException:
    """ForbiddenException 测试"""

    def test_status_code_403(self):
        """测试状态码为 403"""
        exc = ForbiddenException()
        assert exc.status_code == 403


class TestBadRequestException:
    """BadRequestException 测试"""

    def test_status_code_400(self):
        """测试状态码为 400"""
        exc = BadRequestException()
        assert exc.status_code == 400


class TestConflictException:
    """ConflictException 测试"""

    def test_status_code_409(self):
        """测试状态码为 409"""
        exc = ConflictException()
        assert exc.status_code == 409


class TestExceptionInheritance:
    """异常继承关系测试"""

    def test_all_exceptions_are_business(self):
        """测试所有异常都是 BusinessException 的子类"""
        assert issubclass(NotFoundException, BusinessException)
        assert issubclass(UnauthorizedException, BusinessException)
        assert issubclass(ForbiddenException, BusinessException)
        assert issubclass(BadRequestException, BusinessException)
        assert issubclass(ConflictException, BusinessException)

    def test_catch_by_parent(self):
        """测试父类可以捕获子类"""
        try:
            raise NotFoundException("找不到了")
        except BusinessException as e:
            assert e.status_code == 404
