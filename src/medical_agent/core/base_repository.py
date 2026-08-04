# -*- coding: utf-8 -*-
# 通用异步CRUD仓库 - 为所有数据模型提供一致的数据库操作接口
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from sqlalchemy import func, select, update as sa_update, delete as sa_delete
from sqlalchemy.ext.asyncio import AsyncSession

from medical_agent.core.base_model import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """通用CRUD仓库基类，提供标准数据库操作方法"""

    def __init__(self, model: Type[T], db: AsyncSession):
        self.model = model
        self.db = db

    async def create(self, **kwargs: Any) -> T:
        """创建新记录"""
        instance = self.model(**kwargs)
        self.db.add(instance)
        await self.db.flush()
        await self.db.refresh(instance)
        return instance

    async def get_by_id(self, obj_id: int) -> Optional[T]:
        """根据主键ID获取记录"""
        result = await self.db.execute(
            select(self.model).where(self.model.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def find_one(self, **filters: Any) -> Optional[T]:
        """根据条件查找单条记录"""
        stmt = select(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def find_all(
        self,
        skip: int = 0,
        limit: int = 100,
        **filters: Any,
    ) -> List[T]:
        """分页查找符合条件的记录列表"""
        stmt = select(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        stmt = stmt.offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count(self, **filters: Any) -> int:
        """统计符合条件的记录数"""
        stmt = select(func.count()).select_from(self.model)
        for key, value in filters.items():
            stmt = stmt.where(getattr(self.model, key) == value)
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    async def update(self, obj_id: int, **updates: Any) -> Optional[T]:
        """更新记录并返回更新后的实例"""
        stmt = (
            sa_update(self.model)
            .where(self.model.id == obj_id)
            .values(**updates)
        )
        await self.db.execute(stmt)
        return await self.get_by_id(obj_id)

    async def delete(self, obj_id: int) -> bool:
        """根据ID删除记录，返回是否成功"""
        stmt = sa_delete(self.model).where(self.model.id == obj_id)
        result = await self.db.execute(stmt)
        return result.rowcount > 0
