# -*- coding: utf-8 -*-
# 认证中间件 - 全局JWT验证与请求状态注入
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from medical_agent.core.security import decode_token

# 跳过认证的路径列表
SKIP_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
}


class AuthMiddleware(BaseHTTPMiddleware):
    """全局认证中间件，提取JWT并附加到request.state"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # 检查是否跳过认证
        path = request.url.path.rstrip("/")
        if path in SKIP_PATHS or path.startswith("/docs") or path.startswith("/redoc"):
            return await call_next(request)

        # 提取Bearer令牌
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                payload = decode_token(token)
                request.state.user = {
                    "user_id": payload["sub"],
                    "role": payload["role"],
                }
            except Exception:
                # 解析失败不阻断请求，由端点级别依赖处理认证
                pass

        return await call_next(request)
