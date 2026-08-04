# -*- coding: utf-8 -*-
# 日志中间件 - 请求日志记录与状态监控
import time

from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class LoggingMiddleware(BaseHTTPMiddleware):
    """HTTP请求日志中间件，记录方法、路径、状态码和耗时"""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # 跳过健康检查路径的日志
        path = request.url.path
        if path.startswith("/health"):
            return await call_next(request)

        start_time = time.time()
        response = await call_next(request)
        duration = (time.time() - start_time) * 1000  # 转换为毫秒

        log_msg = f"{request.method} {path} -> {response.status_code} ({duration:.1f}ms)"

        status = response.status_code
        if 400 <= status < 500:
            logger.warning(log_msg)
        elif status >= 500:
            logger.error(log_msg)
        else:
            logger.info(log_msg)

        return response
