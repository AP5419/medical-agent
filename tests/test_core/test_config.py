# -*- coding: utf-8 -*-
# 配置模块测试
import pytest

from medical_agent.core.config import get_settings


class TestConfig:
    """配置加载测试"""

    def test_config_singleton(self):
        """测试配置单例模式"""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_default_values(self):
        """测试默认配置值"""
        settings = get_settings()
        assert settings.APP_NAME == "medical-agent"
        assert settings.APP_ENV == "dev"

    def test_database_url_contains_mysql_aiomysql(self):
        """测试 DATABASE_URL 包含 mysql+aiomysql"""
        settings = get_settings()
        url = settings.DATABASE_URL
        assert "mysql+aiomysql" in url
        assert str(settings.DB_PORT) in url

    def test_db_port_default(self):
        """测试 DB_PORT 类默认值（.env 未覆盖时）"""
        from medical_agent.core.config import Settings
        assert Settings.model_fields["DB_PORT"].default == 15308

    def test_database_sync_url_contains_pymysql(self):
        """测试 DATABASE_SYNC_URL 包含 mysql+pymysql"""
        settings = get_settings()
        url = settings.DATABASE_SYNC_URL
        assert "mysql+pymysql" in url

    def test_neo4j_default_uri(self):
        """测试 Neo4j 默认 URI"""
        settings = get_settings()
        assert "bolt://" in settings.NEO4J_URI

    def test_milvus_default_port(self):
        """测试 Milvus 默认端口"""
        settings = get_settings()
        assert settings.MILVUS_PORT == 19530
