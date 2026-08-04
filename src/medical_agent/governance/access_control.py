# -*- coding: utf-8 -*-
# RBAC访问控制模块 - 基于角色的权限矩阵管理
from enum import Enum
from typing import Dict, List

from fastapi import Depends, HTTPException, status


class UserRole(str, Enum):
    """用户角色枚举"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    PHARMACIST = "pharmacist"
    ADMIN = "admin"


class Resource(str, Enum):
    """资源枚举"""
    CHAT = "chat"
    INQUIRY = "inquiry"
    REPORT = "report"
    REPORT_UPLOAD = "report_upload"
    DRUG = "drug"
    PRESCRIPTION = "prescription"
    KNOWLEDGE = "knowledge"
    CLINICAL_GUIDELINE = "clinical_guideline"
    OPERATION_DATA = "operation_data"
    STATISTICS = "statistics"
    USER_MANAGEMENT = "user_management"
    AUDIT_LOG = "audit_log"
    SYSTEM_CONFIG = "system_config"


class Action(str, Enum):
    """操作枚举"""
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    EXPORT = "export"


# 权限矩阵：角色 -> {资源: [允许的操作]}
PERMISSION_MATRIX: Dict[str, Dict[str, List[str]]] = {
    UserRole.PATIENT: {
        Resource.CHAT: [Action.READ, Action.CREATE],
        Resource.INQUIRY: [Action.READ, Action.CREATE],
        Resource.REPORT: [Action.READ, Action.CREATE],
        Resource.REPORT_UPLOAD: [Action.CREATE],
        Resource.DRUG: [Action.READ],
        Resource.KNOWLEDGE: [Action.READ],
    },
    UserRole.DOCTOR: {
        Resource.CHAT: [Action.READ, Action.CREATE],
        Resource.INQUIRY: [Action.READ, Action.CREATE],
        Resource.REPORT: [Action.READ, Action.CREATE],
        Resource.REPORT_UPLOAD: [Action.CREATE],
        Resource.DRUG: [Action.READ, Action.CREATE],
        Resource.PRESCRIPTION: [Action.READ, Action.CREATE, Action.UPDATE],
        Resource.KNOWLEDGE: [Action.READ],
        Resource.CLINICAL_GUIDELINE: [Action.READ],
        Resource.OPERATION_DATA: [Action.READ],
    },
    UserRole.ADMIN: {
        Resource.CHAT: [Action.READ],
        Resource.INQUIRY: [Action.READ],
        Resource.REPORT: [Action.READ],
        Resource.DRUG: [Action.READ, Action.CREATE, Action.UPDATE, Action.DELETE],
        Resource.PRESCRIPTION: [Action.READ],
        Resource.KNOWLEDGE: [Action.READ, Action.CREATE, Action.UPDATE, Action.DELETE],
        Resource.CLINICAL_GUIDELINE: [Action.READ, Action.CREATE, Action.UPDATE, Action.DELETE],
        Resource.OPERATION_DATA: [Action.READ, Action.CREATE, Action.UPDATE, Action.DELETE],
        Resource.STATISTICS: [Action.READ, Action.EXPORT],
        Resource.USER_MANAGEMENT: [Action.READ, Action.CREATE, Action.UPDATE, Action.DELETE],
        Resource.AUDIT_LOG: [Action.READ, Action.EXPORT],
        Resource.SYSTEM_CONFIG: [Action.READ, Action.UPDATE],
    },
}


class AccessControl:
    """访问控制类，基于权限矩阵进行权限校验"""

    def check_permission(self, user_role: str, resource: str, action: str) -> bool:
        """检查用户角色是否对指定资源有指定操作权限"""
        # 验证输入的有效性
        try:
            valid_role = UserRole(user_role)
            valid_resource = Resource(resource)
            valid_action = Action(action)
        except ValueError:
            return False

        role_permissions = PERMISSION_MATRIX.get(valid_role.value, {})
        allowed_actions = role_permissions.get(valid_resource.value, [])
        return valid_action.value in allowed_actions

    def has_any_permission(self, user_role: str, resources: List[str], action: str) -> bool:
        """检查用户角色对任一资源是否有指定操作权限"""
        for resource in resources:
            if self.check_permission(user_role, resource, action):
                return True
        return False

    def has_all_permissions(
        self, user_role: str, required_permissions: List[tuple]
    ) -> bool:
        """检查用户角色是否拥有所有指定权限

        Args:
            user_role: 用户角色
            required_permissions: 权限列表，格式 [(resource, action), ...]

        Returns:
            bool: 是否拥有所有权限
        """
        for resource, action in required_permissions:
            if not self.check_permission(user_role, resource, action):
                return False
        return True


# 访问控制单例
_access_control: AccessControl = None


def get_access_control() -> AccessControl:
    """获取AccessControl单例"""
    global _access_control
    if _access_control is None:
        _access_control = AccessControl()
    return _access_control


def require_permission(resource: str, action: str):
    """FastAPI权限依赖工厂，用于接口级别权限校验"""
    from medical_agent.api.deps import get_current_user

    async def _check(user: dict = Depends(get_current_user)):
        ac = get_access_control()
        if not ac.check_permission(user["role"], resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"无权限执行 {action} 操作于 {resource}",
            )
        return user

    return _check
