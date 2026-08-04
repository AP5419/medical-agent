# -*- coding: utf-8 -*-
"""
医疗多智能体系统 - FastAPI 应用入口
架构: 8层架构（mg文档 SVG 架构图）
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from sqlalchemy import text
from loguru import logger

from medical_agent.core.config import get_settings
from medical_agent.core.exceptions import register_exception_handlers
from medical_agent.core.logger import setup_logger

# Layer 8: 基础设施层
from medical_agent.infra.mysql import engine
from medical_agent.infra.redis import get_redis_client, close_redis_pools
from medical_agent.infra.milvus import check_milvus_health, close_milvus_client
from medical_agent.infra.neo4j import check_neo4j_health, close_neo4j_driver, get_neo4j_driver
from medical_agent.infra.minio import ensure_bucket_exists, get_minio_client

# Layer 2: 接口层路由
from medical_agent.api.routers.chat import router as chat_router
from medical_agent.api.routers.auth import router as auth_router
from medical_agent.api.routers.upload import router as upload_router

# 中间件
from medical_agent.middleware.logging import LoggingMiddleware
from medical_agent.middleware.auth import AuthMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理: 启动时连接基础设施, 关闭时释放资源"""
    setup_logger()
    settings = get_settings()
    logger.info(f"{settings.APP_NAME} 启动中... | 环境: {settings.APP_ENV}")

    try:
        ensure_bucket_exists()
        logger.info("MinIO 桶已就绪")
    except Exception as e:
        logger.error(f"MinIO 桶初始化失败: {e}")

    try:
        if check_milvus_health():
            logger.info("Milvus 连接正常")
    except Exception as e:
        logger.error(f"Milvus 连接失败: {e}")

    try:
        if await check_neo4j_health():
            logger.info("Neo4j 连接正常")
    except Exception as e:
        logger.error(f"Neo4j 连接失败: {e}")

    logger.info(f"{settings.APP_NAME} 启动完成")
    yield
    logger.info(f"{settings.APP_NAME} 关闭中...")
    await close_neo4j_driver()
    close_milvus_client()
    await close_redis_pools()
    await engine.dispose()
    logger.info(f"{settings.APP_NAME} 已关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        debug=settings.APP_DEBUG,
        lifespan=lifespan,
        description="灵枢医疗多智能体系统 - 基于8层架构的医疗AI助手",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:7860"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(AuthMiddleware)
    register_exception_handlers(app)

    app.include_router(auth_router)
    app.include_router(chat_router)
    app.include_router(upload_router)

    return app


app = create_app()


@app.get("/")
async def root():
    """重定向到 API 文档"""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/health/deps")
async def health_deps():
    """核心依赖健康检查: MySQL / Redis / MinIO / Milvus / Neo4j"""
    settings = get_settings()
    result = {"status": "ok", "dependencies": {}}

    deps = ["mysql", "redis", "minio", "milvus", "neo4j"]
    for d in deps:
        result["dependencies"][d] = {"ok": True, "error": ""}

    # MySQL
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.commit()
    except Exception as e:
        result["dependencies"]["mysql"] = {"ok": False, "error": str(e)}

    # Redis
    try:
        r = await get_redis_client()
        await r.ping()
    except Exception as e:
        result["dependencies"]["redis"] = {"ok": False, "error": str(e)}

    # MinIO
    try:
        get_minio_client().bucket_exists(settings.MINIO_BUCKET)
    except Exception as e:
        result["dependencies"]["minio"] = {"ok": False, "error": str(e)}

    # Milvus
    try:
        if not check_milvus_health():
            raise RuntimeError("Milvus 连接检查失败")
    except Exception as e:
        result["dependencies"]["milvus"] = {"ok": False, "error": str(e)}

    # Neo4j
    try:
        await get_neo4j_driver().verify_connectivity()
    except Exception as e:
        result["dependencies"]["neo4j"] = {"ok": False, "error": str(e)}

    if any(not v["ok"] for v in result["dependencies"].values()):
        result["status"] = "degraded"

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("medical_agent.main:app", host="0.0.0.0", port=8080, reload=True)
