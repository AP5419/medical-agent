# -*- coding: utf-8 -*-
# 业务异常类与FastAPI全局异常处理器
from typing import Union

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class BusinessException(Exception):
    """业务异常基类"""

    def __init__(self, message: str, status_code: int = 400, code: Union[str, int] = "400"):
        self.message = message
        self.status_code = status_code
        self.code = code
        super().__init__(self.message)


class NotFoundException(BusinessException):
    """资源未找到异常 (404)"""

    def __init__(self, message: str = "资源未找到", code: Union[str, int] = "404"):
        super().__init__(message=message, status_code=404, code=code)


class UnauthorizedException(BusinessException):
    """未授权异常 (401)"""

    def __init__(self, message: str = "未授权访问", code: Union[str, int] = "401"):
        super().__init__(message=message, status_code=401, code=code)


class ForbiddenException(BusinessException):
    """禁止访问异常 (403)"""

    def __init__(self, message: str = "禁止访问", code: Union[str, int] = "403"):
        super().__init__(message=message, status_code=403, code=code)


class BadRequestException(BusinessException):
    """请求参数错误异常 (400)"""

    def __init__(self, message: str = "请求参数错误", code: Union[str, int] = "400"):
        super().__init__(message=message, status_code=400, code=code)


class ConflictException(BusinessException):
    """资源冲突异常 (409)"""

    def __init__(self, message: str = "资源冲突", code: Union[str, int] = "409"):
        super().__init__(message=message, status_code=409, code=code)


def register_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器到FastAPI应用"""

    @app.exception_handler(BusinessException)
    async def business_exception_handler(request: Request, exc: BusinessException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "code": str(exc.code),
                "message": exc.message,
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "code": "404",
                "message": "请求的资源不存在",
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "code": "500",
                "message": "服务器内部错误",
            },
        )
