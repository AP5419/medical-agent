# -*- coding: utf-8 -*-
# 数据脱敏模块 - 敏感信息遮蔽处理
import re
from typing import Any


def desensitize_phone(phone: str) -> str:
    """手机号脱敏：保留前3位和后4位，中间用*替换"""
    if not phone or not isinstance(phone, str):
        return phone
    phone = phone.strip()
    if len(phone) < 7:
        return "*" * len(phone) if len(phone) > 0 else phone
    return phone[:3] + "****" + phone[-4:]


def desensitize_id_card(id_card: str) -> str:
    """身份证号脱敏：保留前3位和后4位"""
    if not id_card or not isinstance(id_card, str):
        return id_card
    id_card = id_card.strip()
    if len(id_card) < 7:
        return "*" * len(id_card) if len(id_card) > 0 else id_card
    return id_card[:3] + "*" * (len(id_card) - 7) + id_card[-4:]


def desensitize_name(name: str) -> str:
    """姓名脱敏：保留首字符，其余用*替换（张*，欧阳**）"""
    if not name or not isinstance(name, str):
        return name
    name = name.strip()
    if len(name) <= 1:
        return name
    return name[0] + "*" * (len(name) - 1)


def desensitize_patient_data(data: dict) -> dict:
    """对患者数据进行字段级脱敏"""
    if not isinstance(data, dict):
        return data

    # 字段名与脱敏函数的映射
    field_mask_map = {
        "phone": desensitize_phone,
        "mobile": desensitize_phone,
        "telephone": desensitize_phone,
        "id_card": desensitize_id_card,
        "id_number": desensitize_id_card,
        "identity": desensitize_id_card,
        "name": desensitize_name,
        "real_name": desensitize_name,
        "patient_name": desensitize_name,
        "contact": desensitize_phone,
    }

    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            key_lower = key.lower()
            # 精确匹配或包含已知敏感字段名
            mask_func = None
            for pattern, func in field_mask_map.items():
                if pattern in key_lower:
                    mask_func = func
                    break
            if mask_func:
                result[key] = mask_func(value)
            else:
                result[key] = value
        elif isinstance(value, dict):
            result[key] = desensitize_patient_data(value)
        elif isinstance(value, list):
            result[key] = [
                desensitize_patient_data(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


# 手机号正则：匹配中国大陆手机号
PHONE_PATTERN = re.compile(r"(?<!\d)(1[3-9]\d)\d{4}(\d{4})(?!\d)")

# 身份证号正则：匹配18位或15位身份证号
ID_CARD_PATTERN = re.compile(r"(?<!\d)(\d{3})\d{9,12}(\d{4})(?!\d)")


def desensitize_text(text: str) -> str:
    """对自由文本中的手机号和身份证号进行正则脱敏"""
    if not text or not isinstance(text, str):
        return text

    # 脱敏手机号
    text = PHONE_PATTERN.sub(r"\1****\2", text)

    # 脱敏身份证号
    def mask_id_card(match: re.Match) -> str:
        full = match.group(0)
        prefix = match.group(1)
        suffix = match.group(2)
        masked_len = len(full) - len(prefix) - len(suffix)
        return prefix + "*" * masked_len + suffix

    text = ID_CARD_PATTERN.sub(mask_id_card, text)

    return text
