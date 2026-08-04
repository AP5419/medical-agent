# -*- coding: utf-8 -*-
# RBAC 访问控制模块测试
import pytest

from medical_agent.governance.access_control import AccessControl


class TestAccessControl:
    """访问控制测试"""

    @pytest.fixture
    def ac(self):
        """创建 AccessControl 实例"""
        return AccessControl()

    # ---------- patient 角色测试 ----------
    def test_patient_can_read_chat(self, ac):
        """患者可以查看对话"""
        assert ac.check_permission("patient", "chat", "read") is True

    def test_patient_can_create_chat(self, ac):
        """患者可以创建对话"""
        assert ac.check_permission("patient", "chat", "create") is True

    def test_patient_cannot_delete_drug(self, ac):
        """患者不能删除药品"""
        assert ac.check_permission("patient", "drug", "delete") is False

    def test_patient_cannot_access_user_management(self, ac):
        """患者不能访问用户管理"""
        assert ac.check_permission("patient", "user_management", "read") is False

    # ---------- doctor 角色测试 ----------
    def test_doctor_can_read_prescription(self, ac):
        """医生可以读取处方"""
        assert ac.check_permission("doctor", "prescription", "read") is True

    def test_doctor_can_create_prescription(self, ac):
        """医生可以创建处方"""
        assert ac.check_permission("doctor", "prescription", "create") is True

    def test_doctor_can_update_prescription(self, ac):
        """医生可以更新处方"""
        assert ac.check_permission("doctor", "prescription", "update") is True

    def test_doctor_cannot_access_audit_log(self, ac):
        """医生不能访问审计日志"""
        assert ac.check_permission("doctor", "audit_log", "read") is False

    def test_doctor_cannot_access_system_config(self, ac):
        """医生不能访问系统配置"""
        assert ac.check_permission("doctor", "system_config", "read") is False

    # ---------- admin 角色测试 ----------
    def test_admin_can_delete_drug(self, ac):
        """管理员可以删除药品"""
        assert ac.check_permission("admin", "drug", "delete") is True

    def test_admin_can_manage_users(self, ac):
        """管理员可以管理用户"""
        assert ac.check_permission("admin", "user_management", "read") is True
        assert ac.check_permission("admin", "user_management", "create") is True
        assert ac.check_permission("admin", "user_management", "update") is True
        assert ac.check_permission("admin", "user_management", "delete") is True

    def test_admin_can_read_audit_log(self, ac):
        """管理员可以查看审计日志"""
        assert ac.check_permission("admin", "audit_log", "read") is True

    def test_admin_can_export_statistics(self, ac):
        """管理员可以导出统计"""
        assert ac.check_permission("admin", "statistics", "export") is True

    # ---------- 无效输入测试 ----------
    def test_invalid_role_returns_false(self, ac):
        """测试无效角色返回 False"""
        assert ac.check_permission("guest", "chat", "read") is False

    def test_invalid_resource_returns_false(self, ac):
        """测试无效资源返回 False"""
        assert ac.check_permission("patient", "nonexistent", "read") is False

    def test_invalid_action_returns_false(self, ac):
        """测试无效操作返回 False"""
        assert ac.check_permission("patient", "chat", "execute") is False
