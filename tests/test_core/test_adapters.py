# -*- coding: utf-8 -*-
# 外部系统适配器测试
import pytest

from medical_agent.adapters.his import HISAdapter
from medical_agent.adapters.lis import LISAdapter
from medical_agent.adapters.emr import EMRAdapter


class TestHISAdapter:
    """HIS 适配器测试"""

    @pytest.mark.asyncio
    async def test_search_drugs_by_name(self):
        """测试按药品名称搜索"""
        results = await HISAdapter.search_drugs("阿莫西林")
        assert len(results) >= 1
        assert results[0]["name"] == "阿莫西林胶囊"

    @pytest.mark.asyncio
    async def test_search_drugs_no_match(self):
        """测试无匹配结果"""
        results = await HISAdapter.search_drugs("不存在的药品XYZ")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_drugs_by_code(self):
        """测试按药品编码搜索"""
        results = await HISAdapter.search_drugs("DRUG001")
        assert len(results) == 1
        assert results[0]["name"] == "阿莫西林胶囊"

    @pytest.mark.asyncio
    async def test_get_drug_by_code(self):
        """测试按编码精确查询"""
        drug = await HISAdapter.get_drug_by_code("DRUG003")
        assert drug is not None
        assert drug["name"] == "奥美拉唑肠溶片"

    @pytest.mark.asyncio
    async def test_get_drug_by_code_not_found(self):
        """测试编码不存在"""
        drug = await HISAdapter.get_drug_by_code("DRUG999")
        assert drug is None

    @pytest.mark.asyncio
    async def test_check_stock(self):
        """测试库存检查"""
        stock = await HISAdapter.check_stock("DRUG001")
        assert stock is not None
        assert stock["stock"] == 500
        assert stock["available"] is True


class TestLISAdapter:
    """LIS 适配器测试"""

    @pytest.mark.asyncio
    async def test_search_reports_by_patient(self):
        """测试按患者姓名查询报告"""
        results = await LISAdapter.search_reports("张三")
        assert len(results) >= 1
        assert results[0]["patient_name"] == "张三"

    @pytest.mark.asyncio
    async def test_search_reports_with_type(self):
        """测试按患者姓名+报告类型查询"""
        results = await LISAdapter.search_reports("张三", "血常规")
        assert len(results) >= 1
        assert results[0]["report_type"] == "血常规"

    @pytest.mark.asyncio
    async def test_search_reports_no_match(self):
        """测试无匹配患者"""
        results = await LISAdapter.search_reports("不存在的患者")
        assert results == []

    @pytest.mark.asyncio
    async def test_get_report_by_id(self):
        """测试按ID查询报告"""
        report = await LISAdapter.get_report_by_id("L20251215001")
        assert report is not None
        assert report["patient_name"] == "张三"

    @pytest.mark.asyncio
    async def test_get_abnormal_indicators(self):
        """测试异常指标提取"""
        report = {
            "indicators": [
                {"name": "白细胞计数", "status": "偏高"},
                {"name": "红细胞计数", "status": "正常"},
            ]
        }
        abnormal = await LISAdapter.get_abnormal_indicators(report)
        assert len(abnormal) == 1
        assert abnormal[0]["name"] == "白细胞计数"


class TestEMRAdapter:
    """EMR 适配器测试"""

    @pytest.mark.asyncio
    async def test_get_patient_history(self):
        """测试查询患者就诊历史"""
        records = await EMRAdapter.get_patient_history("张三")
        assert len(records) >= 1
        assert records[0]["patient_name"] == "张三"

    @pytest.mark.asyncio
    async def test_get_patient_history_no_match(self):
        """测试无匹配患者"""
        records = await EMRAdapter.get_patient_history("不存在的患者")
        assert records == []

    @pytest.mark.asyncio
    async def test_get_patient_summary(self):
        """测试患者就诊摘要"""
        summary = await EMRAdapter.get_patient_summary("张三")
        assert summary["total_visits"] >= 1
        assert summary["last_visit"] is not None
        assert "date" in summary["last_visit"]
        assert "department" in summary["last_visit"]
        assert len(summary["diagnoses"]) >= 1

    @pytest.mark.asyncio
    async def test_get_patient_summary_no_data(self):
        """测试无数据患者的摘要"""
        summary = await EMRAdapter.get_patient_summary("王五")
        assert summary["total_visits"] == 0
        assert summary["last_visit"] is None
        assert summary["diagnoses"] == []
        assert summary["drugs_used"] == []

    @pytest.mark.asyncio
    async def test_get_record_by_id(self):
        """测试按ID查询记录"""
        record = await EMRAdapter.get_record_by_id("EMR202501001")
        assert record is not None
        assert record["patient_name"] == "张三"

    @pytest.mark.asyncio
    async def test_get_record_by_id_not_found(self):
        """测试ID不存在"""
        record = await EMRAdapter.get_record_by_id("EMR999999")
        assert record is None


class TestPACSAdapter:
    """PACS 适配器测试"""

    @pytest.mark.asyncio
    async def test_search_studies_by_patient(self):
        """测试按患者姓名查询影像检查"""
        from medical_agent.adapters.pacs import PACSAdapter
        studies = await PACSAdapter().search_studies(patient_name="张三")
        assert len(studies) >= 1
        assert studies[0]["patient_name"] == "张三"

    @pytest.mark.asyncio
    async def test_search_studies_by_modality(self):
        """测试按检查类型筛选"""
        from medical_agent.adapters.pacs import PACSAdapter
        studies = await PACSAdapter().search_studies(modality="CT")
        assert len(studies) >= 1
        for s in studies:
            assert s["modality"] == "CT"

    @pytest.mark.asyncio
    async def test_get_study_by_id(self):
        """测试按ID获取检查记录"""
        from medical_agent.adapters.pacs import PACSAdapter
        study = await PACSAdapter().get_study_by_id("IMG20240701001")
        assert study is not None
        assert study["modality"] == "CT"

    @pytest.mark.asyncio
    async def test_get_imaging_report(self):
        """测试获取影像报告"""
        from medical_agent.adapters.pacs import PACSAdapter
        report = await PACSAdapter().get_imaging_report("IMG20240701002")
        assert report is not None
        assert "findings" in report
        assert "impression" in report

    @pytest.mark.asyncio
    async def test_get_patient_imaging_summary(self):
        """测试患者影像摘要"""
        from medical_agent.adapters.pacs import PACSAdapter
        summary = await PACSAdapter().get_patient_imaging_summary("张三")
        assert summary["total_studies"] >= 1
        assert len(summary["studies"]) >= 1

    @pytest.mark.asyncio
    async def test_get_modality_statistics(self):
        """测试检查类型统计"""
        from medical_agent.adapters.pacs import PACSAdapter
        stats = await PACSAdapter().get_modality_statistics()
        assert len(stats) > 0
        modalities = [s["modality"] for s in stats]
        assert "CT" in modalities
