# -*- coding: utf-8 -*-
"""
MS GraphRAG 索引初始化脚本
功能: 从 data/documents/ 目录中的医疗文档构建知识图谱索引
用法: python scripts/init_graphrag.py
前置条件: 
  1. pip install graphrag
  2. 配置 settings.yaml 中的 LLM API Key
  3. 文档已放入 data/documents/ 子目录


1. PDF 预处理 → MinerU 把 PDF 转成 .txt
2. 统计全部 .txt 文件（含刚生成的）
3. graphrag index 索引所有文件
"""

import subprocess
import sys
import os
import asyncio
from pathlib import Path

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from medical_agent.core.config import get_settings


async def init_graphrag_index():
    """初始化 GraphRAG 索引"""
    root_dir = Path(__file__).resolve().parent.parent
    
    logger.info(f"项目根目录: {root_dir}")
    
    # 检查文档目录
    doc_dir = root_dir / "data" / "documents"
    if not doc_dir.exists():
        logger.error(f"文档目录不存在: {doc_dir}")
        sys.exit(1)

    # ── PDF 预处理：MinerU 解析 → .txt → graphrag 可索引 ──
    pdf_dir = doc_dir / "pdf"
    pdf_files = list(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    if pdf_files:
        logger.info(f"发现 {len(pdf_files)} 个 PDF 文件，使用 MinerU 解析...")
        try:
            from medical_agent.engines.rag.mineru_client import MinerUClient
            mineru = MinerUClient()
            for pdf_path in pdf_files:
                txt_path = pdf_path.with_suffix(".txt")
                if txt_path.exists():
                    logger.info(f"  已存在，跳过: {pdf_path.name}")
                    continue
                logger.info(f"  解析: {pdf_path.name}...")
                result = await mineru.parse_file(str(pdf_path))
                if result["success"]:
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(result["markdown"])
                    logger.info(f"    → {txt_path.name} ({len(result['markdown'])} 字符)")
                else:
                    logger.warning(f"  解析失败: {pdf_path.name} - {result['error']}")
        except ImportError:
            logger.warning("MinerU 未安装，跳过 PDF 预处理 (pip install mineru)")
        except Exception as e:
            logger.warning(f"PDF 预处理异常: {e}")
    else:
        logger.info("未发现 PDF 文件，跳过预处理")

    # 统计文档数量（PDF 预处理后可能有新的 .txt 文件）
    txt_files = list(doc_dir.rglob("*.txt"))
    logger.info(f"发现 {len(txt_files)} 个文本文档:")
    for f in txt_files:
        logger.info(f"  - {f.relative_to(root_dir)}")

    if len(txt_files) == 0:
        logger.error("没有找到文档，请先将文档放入 data/documents/ 目录")
        sys.exit(1)
    
    # 检查 graphrag 是否安装
    try:
        import graphrag
        try:
            logger.info(f"graphrag 版本: {graphrag.__version__}")
        except AttributeError:
            logger.info("graphrag 已安装（包未提供 __version__ 属性）")
    except ImportError:
        logger.error("graphrag 未安装，执行: pip install graphrag")
        sys.exit(1)
    
    # 检查 settings.yaml
    settings_path = root_dir / "settings.yaml"
    if not settings_path.exists():
        logger.error(f"settings.yaml 不存在: {settings_path}")
        sys.exit(1)
    
    # 设置环境变量
    env = os.environ.copy()
    settings = get_settings()
    # 从 Python settings 注入 .env 中的值到子进程（仅当 .env 有值时覆盖，否则保留 OS 环境变量）
    if settings.DASHSCOPE_API_KEY:
        env["DASHSCOPE_API_KEY"] = settings.DASHSCOPE_API_KEY
    if settings.BASE_URL_CHAT:
        env["BASE_URL_CHAT"] = settings.BASE_URL_CHAT
    env["PYTHONIOENCODING"] = "utf-8"

    # 运行 graphrag index
    logger.info("开始构建索引... (这可能需要几分钟，取决于文档数量和 LLM 速度)")
    
    try:
        # 直接输出到终端，用户可看到实时进度
        result = subprocess.run(
            ["graphrag", "index", "--root", str(root_dir)],
            cwd=str(root_dir),
            env=env,
        )

        if result.returncode == 0:
            logger.info("索引构建成功!")
            output_dir = root_dir / "output"
            if output_dir.exists():
                parquet_files = list(output_dir.rglob("*.parquet"))
                logger.info(f"生成 {len(parquet_files)} 个 Parquet 文件")
                for pf in parquet_files:
                    logger.info(f"  - {pf.name}")
        else:
            logger.error(f"索引构建失败 (code={result.returncode})")
    except FileNotFoundError:
        logger.error("未找到 graphrag 命令，请确保 pip install graphrag 后 PATH 包含 graphrag")
    except Exception as e:
        logger.error(f"索引构建异常: {e}")


if __name__ == "__main__":
    from core.logger import setup_logger
    setup_logger()
    asyncio.run(init_graphrag_index())
