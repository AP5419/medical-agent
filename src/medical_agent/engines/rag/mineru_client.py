# -*- coding: utf-8 -*-
"""
Layer 5: 能力引擎层
模块: MinerU 文档解析客户端
技术栈: OpenDataLab/MinerU 官方包
职责: PDF/DOCX/图片 → Markdown 结构化解析
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

from loguru import logger

from medical_agent.core.config import get_settings


class MinerUClient:
    """
    MinerU 文档解析客户端
    
    医疗场景用途:
        1. PDF 药品说明书 → 结构化 Markdown 文本
        2. 扫描版检验报告 → OCR + 文本提取
        3. DOCX 临床指南 → 保留格式的文本
    
    调用方式:
        - CLI模式: subprocess 调用 mineru 命令（稳定）
        - API模式: 通过 MINERU_API_URL 调用远程服务（可选）
    
    Attributes:
        backend: 解析后端 (pipeline / vlm-engine / hybrid-engine)
        timeout: 单文档超时秒数
    """
    
    def __init__(self, backend: str = "pipeline", timeout: int = 300):
        self.settings = get_settings()
        self.backend = self.settings.MINERU_BACKEND or backend
        self.timeout = self.settings.MINERU_TIMEOUT or timeout
        self.api_url = self.settings.MINERU_API_URL
        logger.info(f"MinerU 客户端已初始化, 后端: {self.backend}")
    
    async def parse_file(self, file_path: str, output_dir: Optional[str] = None) -> dict:
        """
        解析文档文件为 Markdown 文本
        
        Args:
            file_path: 文档路径 (支持 PDF/DOCX/PPTX/XLSX/图片)
            output_dir: 输出目录 (默认: 同目录下的 mineru_output/)
        
        Returns:
            {
                "success": bool,
                "markdown": str,        # 解析后的Markdown文本
                "output_dir": str,      # 输出目录
                "error": str            # 错误信息（如有）
            }
        """
        file_path = os.path.abspath(file_path)
        
        if not os.path.exists(file_path):
            return {"success": False, "markdown": "", "output_dir": "", "error": f"文件不存在: {file_path}"}
        
        if output_dir is None:
            output_dir = str(Path(file_path).parent / "mineru_output")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 本地 CLI 模式（同步，Windows 稳定）
        return self._parse_via_cli(file_path, output_dir)

    def _parse_via_cli(self, file_path: str, output_dir: str) -> dict:
        """通过本地 mineru CLI 同步解析（不捕获输出，进度实时可见）"""
        try:
            cmd = [
                "mineru",
                "-p", file_path,
                "-o", output_dir,
                "-b", self.backend,
            ]

            logger.info(f"MinerU CLI 解析: {' '.join(cmd)}")
            print()  # 分隔线，让 mineru 输出更清晰

            # 不 capture_output，不 timeout——让 mineru 直接输出到终端
            # MinerU 3.x 首次运行需下载模型，可能耗时较长
            result = subprocess.run(cmd, cwd=os.path.dirname(file_path))
            print()

            if result.returncode != 0:
                return {"success": False, "markdown": "", "output_dir": output_dir,
                        "error": f"MinerU 返回非零退出码: {result.returncode}"}

            # 查找输出的 Markdown 文件
            markdown_content = self._find_markdown_output(file_path, output_dir)

            if markdown_content:
                logger.info(f"MinerU 解析成功: {len(markdown_content)} 字符")
                return {"success": True, "markdown": markdown_content, "output_dir": output_dir, "error": ""}
            else:
                return {"success": False, "markdown": "", "output_dir": output_dir, "error": "未找到输出 Markdown 文件"}

        except FileNotFoundError:
            return {"success": False, "markdown": "", "output_dir": output_dir,
                    "error": "mineru 未安装，请执行: pip install mineru"}
        except Exception as e:
            logger.error(f"MinerU 解析异常: {e}")
            return {"success": False, "markdown": "", "output_dir": output_dir, "error": str(e)}
    
    async def _parse_via_api(self, file_path: str, output_dir: str) -> dict:
        """通过远程 MinerU API 解析"""
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                data = aiohttp.FormData()
                data.add_field("file", open(file_path, "rb"), filename=os.path.basename(file_path))
                data.add_field("backend", self.backend)
                
                async with session.post(
                    f"{self.api_url}/file_parse",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        markdown = result.get("content", "")
                        # 保存到输出目录
                        output_file = Path(output_dir) / f"{Path(file_path).stem}.md"
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(markdown)
                        return {"success": True, "markdown": markdown, "output_dir": output_dir, "error": ""}
                    else:
                        return {"success": False, "markdown": "", "output_dir": output_dir, "error": f"API 返回 {resp.status}"}
        except Exception as e:
            return {"success": False, "markdown": "", "output_dir": output_dir, "error": str(e)}
    
    def _find_markdown_output(self, file_path: str, output_dir: str) -> Optional[str]:
        """在 mineru 输出目录中查找对应的 Markdown 文件"""
        file_stem = Path(file_path).stem
        
        # mineru 输出的目录结构: output_dir/文件名/版本号/文件名.md
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith(".md") and file_stem in f:
                    full_path = os.path.join(root, f)
                    try:
                        with open(full_path, "r", encoding="utf-8") as fp:
                            return fp.read()
                    except Exception:
                        pass
        return None
    
    def parse_sync(self, file_path: str, output_dir: Optional[str] = None) -> dict:
        """同步解析（便捷方法）"""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.parse_file(file_path, output_dir))
        finally:
            loop.close()
