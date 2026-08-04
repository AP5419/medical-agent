# -*- coding: utf-8 -*-
"""VLM视觉语言模型客户端 - 医学影像分析、报告OCR、处方识别"""

import base64
import warnings
from typing import Optional

from medical_agent.core.config import get_settings

try:
    from dashscope import MultiModalConversation
    _DASHSCOPE_VL_AVAILABLE = True
except ImportError:
    _DASHSCOPE_VL_AVAILABLE = False
    warnings.warn("dashscope MultiModalConversation 不可用，VLM功能将受限")


class VLMClient:
    """视觉语言模型客户端，支持医学影像分析、化验单OCR、处方识别等"""

    def __init__(self):
        """初始化VLM客户端，加载配置"""
        self.settings = get_settings()
        self.model = self.settings.VL_MODEL

    def _encode_image(self, image_path: str) -> str:
        """将图片文件编码为base64字符串

        Args:
            image_path: 图片文件路径

        Returns:
            base64编码的图片字符串
        """
        with open(image_path, "rb") as f:
            image_data = f.read()
        return base64.b64encode(image_data).decode("utf-8")

    def _build_image_content(self, image_path: str) -> dict:
        """构建多模态对话的图片内容项

        Args:
            image_path: 图片文件路径

        Returns:
            格式化为 data URI 的图片内容字典
        """
        base64_str = self._encode_image(image_path)
        return {"image": f"data:image/jpeg;base64,{base64_str}"}

    def analyze_medical_image(
        self, image_path: str, prompt: str = ""
    ) -> Optional[str]:
        """通用的医学图像分析接口

        Args:
            image_path: 图片路径
            prompt: 分析提示词

        Returns:
            模型分析结果文本
        """
        if not _DASHSCOPE_VL_AVAILABLE:
            return "VLM功能不可用：dashscope未安装"

        if not prompt:
            prompt = "请用中文详细描述这张医学图片的内容，包括关键发现和异常征象。"

        image_content = self._build_image_content(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    image_content,
                    {"text": prompt},
                ],
            }
        ]

        try:
            response = MultiModalConversation.call(model=self.model, messages=messages)
            if response.output and response.output.choices:
                content_list = response.output.choices[0].message.content
                if isinstance(content_list, list):
                    return "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content_list
                    )
                return str(content_list)
            return None
        except Exception as e:
            return f"医学图像分析失败：{str(e)}"

    def extract_report_text(self, image_path: str) -> Optional[str]:
        """从医疗化验报告中提取文字内容（OCR专用）

        Args:
            image_path: 化验单/报告图片路径

        Returns:
            提取的报告文本
        """
        prompt = """请作为OCR识别引擎，精确识别并输出这张医疗化验报告中的所有文字内容。
要求：
1. 保留原始格式和数值，包括检验项目名称、结果值、参考范围、单位
2. 标记异常值（超过参考范围的用【异常】标记）
3. 仅输出识别到的内容，不要添加额外解释"""
        return self.analyze_medical_image(image_path, prompt)

    def analyze_xray(self, image_path: str) -> Optional[str]:
        """从放射科影像医师角度分析X光/CT/MRI影像

        Args:
            image_path: 医学影像文件路径

        Returns:
            影像分析报告
        """
        prompt = """你是一位资深放射科医师。请从专业角度分析这张医学影像（X光/CT/MRI），
包括以下方面：
1. 影像部位和检查类型判断
2. 关键解剖结构描述
3. 异常征象发现（位置、大小、形态、密度）
4. 可能的鉴别诊断方向
5. 建议的进一步检查

请使用专业但易懂的中文描述。注意：此为AI辅助分析，不能替代专业医师诊断。"""
        return self.analyze_medical_image(image_path, prompt)

    def analyze_prescription(self, image_path: str) -> Optional[str]:
        """识别处方笺内容

        Args:
            image_path: 处方笺图片路径

        Returns:
            处方识别结果
        """
        prompt = """请作为OCR引擎识别并结构化输出这份处方笺的内容。
请按以下格式输出：
- 患者姓名：
- 性别/年龄：
- 诊断：
- 药品清单（药品名、规格、用法用量）：
- 医师签名：
- 日期：
- 其他备注：

仅输出识别到的内容，保留原文字。"""
        return self.analyze_medical_image(image_path, prompt)
