# -*- coding: utf-8 -*-
# pytest 全局配置 - 异步事件循环 fixture
import pytest_asyncio


@pytest_asyncio.fixture(scope="session")
def event_loop():
    """为 asyncio 提供会话级事件循环"""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
