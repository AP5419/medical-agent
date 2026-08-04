# -*- coding: utf-8 -*-
"""评估模块公共配置"""
import json
import os
from pathlib import Path
import pytest

DATASET_DIR = Path(__file__).parent / "datasets"

def load_dataset(filename: str) -> list:
    """加载JSON评估数据集"""
    filepath = DATASET_DIR / filename
    if not filepath.exists():
        pytest.skip(f"数据集不存在: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
