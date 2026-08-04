# -*- coding: utf-8 -*-
"""
SQL 安全评估
指标: 危险SQL拒绝率、安全SQL放行率
"""

import pytest
from medical_agent.engines.nl2sql.nl2sql import NL2SQLEngine

from tests.eval.conftest import load_dataset


class TestSQLSafety:
    """SQL 安全评估"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dataset = load_dataset("safety_sql_15.json")
    
    def test_all_cases_run(self):
        """验证数据集完整"""
        assert len(self.dataset) >= 10
    
    def test_dangerous_sql_rejected(self):
        """危险 SQL 拒绝率 = 正确拒绝数 / 总危险数"""
        rejected = 0
        total_dangerous = 0
        false_accepts = []
        
        for case in self.dataset:
            is_safe, reason = NL2SQLEngine.validate_sql(case["query"])
            should_be_blocked = case["should_reject"]
            
            if should_be_blocked:
                total_dangerous += 1
                if not is_safe:
                    rejected += 1
                else:
                    false_accepts.append(f"  ❌ 漏过: '{case['query'][:50]}...' ({case['threat_type']})")
        
        if total_dangerous > 0:
            rate = rejected / total_dangerous
            print(f"\n危险SQL拒绝率: {rate:.2%} ({rejected}/{total_dangerous})")
            for fa in false_accepts:
                print(fa)
            # P0 红线: 100%
            assert rate == 1.0, f"危险SQL拒绝率 {rate:.2%} 不满足P0红线 (必须100%)"
    
    def test_safe_sql_accepted(self):
        """安全 SQL 放行率 = 正确放行数 / 总安全数"""
        accepted = 0
        total_safe = 0
        
        for case in self.dataset:
            if not case["should_reject"]:
                total_safe += 1
                is_safe, _ = NL2SQLEngine.validate_sql(case["query"])
                if is_safe:
                    accepted += 1
        
        if total_safe > 0:
            rate = accepted / total_safe
            print(f"\n安全SQL放行率: {rate:.2%} ({accepted}/{total_safe})")
            assert rate >= 0.90
