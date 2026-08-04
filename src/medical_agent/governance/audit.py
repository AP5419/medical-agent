# -*- coding: utf-8 -*-
# 审计日志模块 - 用户操作记录与追溯
import hashlib
from datetime import datetime
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# 审计日志表的建表SQL
CREATE_AUDIT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '主键ID',
    user_id BIGINT NOT NULL COMMENT '操作用户ID',
    query TEXT NOT NULL COMMENT '用户查询内容',
    agent VARCHAR(100) DEFAULT NULL COMMENT '响应的智能体名称',
    result_summary TEXT DEFAULT NULL COMMENT '结果摘要',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    INDEX idx_audit_user_id (user_id),
    INDEX idx_audit_agent (agent),
    INDEX idx_audit_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计日志表';
"""


class AuditLogger:
    """审计日志记录器，用于记录用户操作和查询历史"""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _ensure_table(self) -> None:
        """确保审计日志表存在"""
        await self.db.execute(text(CREATE_AUDIT_TABLE_SQL))
        await self.db.flush()

    async def log(
        self,
        user_id: int,
        query: str,
        agent: Optional[str] = None,
        result_summary: Optional[str] = None,
    ) -> None:
        """记录一条审计日志"""
        await self._ensure_table()

        insert_sql = text(
            "INSERT INTO audit_logs (user_id, query, agent, result_summary) "
            "VALUES (:user_id, :query, :agent, :result_summary)"
        )
        await self.db.execute(
            insert_sql,
            {
                "user_id": user_id,
                "query": query,
                "agent": agent,
                "result_summary": result_summary,
            },
        )
        await self.db.flush()

    async def query_history(
        self,
        user_id: Optional[int] = None,
        agent: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict]:
        """查询审计日志历史记录"""
        await self._ensure_table()

        conditions = []
        params = {}

        if user_id is not None:
            conditions.append("user_id = :user_id")
            params["user_id"] = user_id

        if agent is not None:
            conditions.append("agent = :agent")
            params["agent"] = agent

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        select_sql = text(
            f"SELECT id, user_id, query, agent, result_summary, created_at "
            f"FROM audit_logs {where_clause} "
            f"ORDER BY created_at DESC LIMIT :limit"
        )
        params["limit"] = limit

        result = await self.db.execute(select_sql, params)
        rows = result.fetchall()

        return [
            {
                "id": row.id,
                "user_id": row.user_id,
                "query": row.query,
                "agent": row.agent,
                "result_summary": row.result_summary,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
