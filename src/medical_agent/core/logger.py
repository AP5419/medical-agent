# -*- coding: utf-8 -*-
# 日志配置模块 - 基于Loguru的统一日志管理
import os
import sys

from loguru import logger

from medical_agent.core.config import get_settings


def setup_logger() -> None:
    """初始化Loguru日志系统"""
    settings = get_settings()

    # 移除默认handler
    logger.remove()

    # 控制台输出 - 彩色格式
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )

    # 确保日志目录存在
    os.makedirs(settings.LOG_DIR, exist_ok=True)

    # 文件日志 - 按天轮转，保留7天
    logger.add(
        os.path.join(settings.LOG_DIR, "medical_agent_{time:YYYY-MM-DD}.log"),
        level=settings.LOG_LEVEL,
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,
    )

    # 错误日志单独文件
    logger.add(
        os.path.join(settings.LOG_DIR, "medical_agent_error_{time:YYYY-MM-DD}.log"),
        level="ERROR",
        rotation="00:00",
        retention="7 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        enqueue=True,
    )

    logger.info("日志系统初始化完成")
