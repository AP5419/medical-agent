# -*- coding: utf-8 -*-
# 统一响应模型 - API接口返回的标准数据结构
from typing import Any, Generic, List, Optional, TypeVar

from pydantic import BaseModel as PydanticBaseModel

T = TypeVar("T")


class ResponseModel(PydanticBaseModel, Generic[T]):
    """统一API响应模型"""

    success: bool = True
    code: str = "0"
    message: str = "ok"
    data: Optional[T] = None


class PageResponseModel(PydanticBaseModel, Generic[T]):
    """分页API响应模型"""

    success: bool = True
    code: str = "0"
    message: str = "ok"
    data: List[T] = []
    total: int = 0
    page: int = 1
    page_size: int = 10
