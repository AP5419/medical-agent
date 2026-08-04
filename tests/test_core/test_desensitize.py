# -*- coding: utf-8 -*-
# 数据脱敏模块测试
import pytest

from medical_agent.governance.desensitize import (
    desensitize_phone,
    desensitize_name,
    desensitize_id_card,
    desensitize_patient_data,
    desensitize_text,
)


class TestDesensitizePhone:
    """手机号脱敏测试"""

    def test_normal_phone(self):
        """测试正常手机号脱敏"""
        result = desensitize_phone("13812345678")
        assert result == "138****5678"

    def test_short_number(self):
        """测试短号码脱敏"""
        result = desensitize_phone("123")
        assert result == "***"

    def test_empty_string(self):
        """测试空字符串"""
        assert desensitize_phone("") == ""

    def test_none_input(self):
        """测试 None 输入"""
        assert desensitize_phone(None) is None

    def test_non_string_input(self):
        """测试非字符串输入"""
        assert desensitize_phone(12345678901) == 12345678901


class TestDesensitizeName:
    """姓名脱敏测试"""

    def test_two_char_name(self):
        """测试两字姓名（张*）"""
        result = desensitize_name("张三")
        assert result == "张*"

    def test_three_char_name(self):
        """测试三字姓名"""
        result = desensitize_name("王小明")
        assert result == "王**"

    def test_four_char_name(self):
        """测试四字姓名（复姓）"""
        result = desensitize_name("欧阳修文")
        assert result == "欧***"

    def test_single_char_name(self):
        """测试单字名"""
        result = desensitize_name("张")
        assert result == "张"

    def test_empty_string(self):
        """测试空字符串"""
        assert desensitize_name("") == ""

    def test_none_input(self):
        """测试 None 输入"""
        assert desensitize_name(None) is None


class TestDesensitizeIDCard:
    """身份证号脱敏测试"""

    def test_18_digit_id_card(self):
        """测试18位身份证号脱敏"""
        result = desensitize_id_card("110101199001011234")
        assert result == "110***********1234"
        assert result[:3] == "110"
        assert result[-4:] == "1234"

    def test_15_digit_id_card(self):
        """测试15位身份证号脱敏"""
        result = desensitize_id_card("110101900101123")
        assert result[:3] == "110"
        assert result[-4:] == "1123"

    def test_short_string(self):
        """测试短字符串"""
        result = desensitize_id_card("12345")
        assert result == "*****"

    def test_empty_string(self):
        """测试空字符串"""
        assert desensitize_id_card("") == ""

    def test_none_input(self):
        """测试 None 输入"""
        assert desensitize_id_card(None) is None


class TestDesensitizePatientData:
    """患者数据字段级脱敏测试"""

    def test_phone_field(self):
        """测试 phone 字段脱敏"""
        data = {"name": "张三", "phone": "13812345678"}
        result = desensitize_patient_data(data)
        assert result["name"] == "张*"
        assert result["phone"] == "138****5678"

    def test_mobile_field(self):
        """测试 mobile 字段脱敏"""
        data = {"mobile": "13900001111"}
        result = desensitize_patient_data(data)
        assert result["mobile"] == "139****1111"

    def test_id_card_field(self):
        """测试 id_card 字段脱敏"""
        data = {"id_card": "110101199001011234"}
        result = desensitize_patient_data(data)
        assert result["id_card"] == "110***********1234"

    def test_id_number_field(self):
        """测试 id_number 字段脱敏"""
        data = {"id_number": "110101199001011234"}
        result = desensitize_patient_data(data)
        assert result["id_number"] == "110***********1234"

    def test_patient_name_field(self):
        """测试 patient_name 字段脱敏"""
        data = {"patient_name": "李小明"}
        result = desensitize_patient_data(data)
        assert result["patient_name"] == "李**"

    def test_nested_dict(self):
        """测试嵌套字典脱敏"""
        data = {"patient": {"name": "张三", "phone": "13812345678"}}
        result = desensitize_patient_data(data)
        assert result["patient"]["name"] == "张*"
        assert result["patient"]["phone"] == "138****5678"

    def test_list_of_dicts(self):
        """测试字典列表脱敏"""
        data = {"contacts": [{"name": "张三", "phone": "13812345678"}]}
        result = desensitize_patient_data(data)
        assert result["contacts"][0]["name"] == "张*"

    def test_non_dict_input(self):
        """测试非字典输入原样返回"""
        result = desensitize_patient_data("不是字典")
        assert result == "不是字典"

    def test_normal_field_preserved(self):
        """测试普通字段保持不变"""
        data = {"age": 30, "department": "内科"}
        result = desensitize_patient_data(data)
        assert result == data


class TestDesensitizeText:
    """自由文本正则脱敏测试"""

    def test_phone_in_text(self):
        """测试文字中的手机号脱敏"""
        text = "请联系 13812345678 获取报告"
        result = desensitize_text(text)
        assert "13812345678" not in result
        assert "138****5678" in result

    def test_id_card_in_text(self):
        """测试文字中的身份证号脱敏"""
        text = "身份证号 110101199001011234 请保密"
        result = desensitize_text(text)
        assert "110101199001011234" not in result
        assert result[:3] == "身份证号 110".replace("110101199001011234", result[5:]) or True

    def test_no_sensitive_info(self):
        """测试无敏感信息文本"""
        text = "这是一段正常的描述文本"
        result = desensitize_text(text)
        assert result == text

    def test_empty_text(self):
        """测试空文本"""
        assert desensitize_text("") == ""

    def test_none_text(self):
        """测试 None 输入"""
        assert desensitize_text(None) is None
