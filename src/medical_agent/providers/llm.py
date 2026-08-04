# -*- coding: utf-8 -*-
"""LLM提供层 - 统一的大语言模型接口（架构第7层：模型层）

模型自动适配规则:
    - 模型名含 deepseek → ChatDeepSeek + DEEPSEEK_API_KEY（需安装 langchain-deepseek）
    - 其他模型（qwen/glm/gpt等）→ ChatOpenAI（OpenAI 兼容协议）
    
DashScope、智谱 GLM、Ollama 等均通过 OpenAI 兼容接口调用。
"""

from functools import lru_cache

from medical_agent.core.config import get_settings
from loguru import logger

# 尝试导入 DeepSeek SDK（可选）
try:
    from langchain_deepseek import ChatDeepSeek
    _DEEPSEEK_AVAILABLE = True
except ImportError:
    _DEEPSEEK_AVAILABLE = False

from langchain_openai import ChatOpenAI


def _use_deepseek(model_name: str) -> bool:
    """检测模型名是否需要使用 DeepSeek SDK"""
    return "deepseek" in model_name.lower() and _DEEPSEEK_AVAILABLE


def get_llm(temperature: float = 0.1):
    """获取 LLM 实例。注：实例创建后缓存在内存中（ChatOpenAI 线程安全）。"""
    return _create_llm(temperature)


@lru_cache(maxsize=4)
def _create_llm(temperature: float = 0.1):
    """创建 LLM 实例（缓存，同温度只创建一次）"""
    settings = get_settings()
    model = settings.CHAT_MODEL

    # DeepSeek 专属通道
    if _use_deepseek(model):
        api_key = getattr(settings, "DEEPSEEK_API_KEY", "") or settings.DASHSCOPE_API_KEY
        return ChatDeepSeek(
            model=model,
            api_key=api_key,
            api_base=settings.BASE_URL_CHAT or "https://api.deepseek.com",
            temperature=temperature,
            max_tokens=2048,
        )

    # OpenAI 兼容通道（DashScope / GLM / Ollama / etc.）
    api_key = settings.DASHSCOPE_API_KEY
    logger.info(f"LLM 初始化: model={model}, base_url={settings.BASE_URL_CHAT}")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=settings.BASE_URL_CHAT,
        temperature=temperature,
        max_tokens=2048,
    )


@lru_cache
def get_llm_qa():
    """获取问答专用 LLM（temperature=0，医疗场景严格模式+关闭 thinking）"""
    return get_llm(temperature=0.0)


@lru_cache
def get_llm_conversation():
    """获取对话专用 LLM（temperature=0.3，轻度灵活）"""
    return get_llm(temperature=0.3)
