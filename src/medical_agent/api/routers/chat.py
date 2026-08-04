# -*- coding: utf-8 -*-
# 对话路由 - 非流式与SSE流式对话
import json
from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from medical_agent.api.deps import get_current_user

router = APIRouter(prefix="/api/v1/chat", tags=["对话"])


class ChatRequest(BaseModel):
    """对话请求"""
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息内容")
    patient_id: Optional[str] = Field(default=None, description="关联的患者ID")


@router.post("/", summary="发送对话消息")
async def chat(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """非流式对话"""
    from medical_agent.orchestration.supervisor import orchestrate
    content = await orchestrate(req.message, req.user_id, req.session_id, current_user["role"])
    return {"content": content, "thread_id": f"{req.user_id}:{req.session_id}"}


async def _stream_generator(req: ChatRequest, role: str):
    """SSE流式生成器（确定性编排，无ReAct）"""
    from medical_agent.orchestration.supervisor import orchestrate

    # 获取完整回复
    try:
        response = await orchestrate(req.message, req.user_id, req.session_id, role)
    except Exception as e:
        yield json.dumps({"event": "error", "data": f"处理异常: {str(e)}"}, ensure_ascii=False) + "\n"
        yield '{"event": "done"}\n'
        return

    # 逐字符流式输出
    for char in response:
        yield json.dumps({"event": "token", "data": char}, ensure_ascii=False) + "\n"
    yield '{"event": "done"}\n'


@router.post("/stream", summary="SSE流式对话")
async def chat_stream(
    req: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    """SSE流式对话"""
    return StreamingResponse(
        _stream_generator(req, current_user["role"]),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
