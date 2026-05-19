"""
解析层：将 PDF / 文本输入转换为 Markdown

支持格式：
- 纯文本描述（如 "roe_pyoy_mrq：单季度ROE同比..."）
- PDF 研报（需要 pymupdf / marker 等库）
- 图片（需要 OCR，暂不支持，建议用户转文本）

输出：标准 Markdown 文本，供下游 Spec 生成使用
"""

import os
from pathlib import Path
from typing import Union


def parse_input(input_path: Union[str, Path]) -> str:
    """
    根据文件后缀自动选择解析方式，返回 Markdown 文本

    Args:
        input_path: 输入文件路径，支持 .txt / .md / .pdf

    Returns:
        解析后的纯文本 / Markdown 字符串
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return _parse_text(input_path)

    if suffix == ".pdf":
        return _parse_pdf(input_path)

    raise ValueError(f"不支持的输入格式: {suffix}，目前仅支持 .txt / .md / .pdf")


def _parse_text(path: Path) -> str:
    """读取文本文件"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _parse_pdf(path: Path) -> str:
    """
    解析 PDF 为 Markdown 文本

    优先尝试 pymupdf，如果没有则尝试 pdfplumber，
    最后降级为提示用户安装依赖
    """
    # 尝试 pymupdf (fitz)
    try:
        import fitz  # noqa

        return _parse_with_pymupdf(path)
    except ImportError:
        pass

    # 尝试 pdfplumber
    try:
        import pdfplumber  # noqa

        return _parse_with_pdfplumber(path)
    except ImportError:
        pass

    raise ImportError(
        "解析 PDF 需要安装依赖: pip install pymupdf 或 pip install pdfplumber"
    )


def _parse_with_pymupdf(path: Path) -> str:
    """使用 PyMuPDF 提取文本和表格"""
    import fitz

    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return "\n\n---\n\n".join(pages)


def _parse_with_pdfplumber(path: Path) -> str:
    """使用 pdfplumber 提取文本"""
    import pdfplumber

    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                texts.append(text.strip())
    return "\n\n---\n\n".join(texts)


# 便捷函数：直接解析文本字符串（无需文件）
def parse_raw_text(text: str) -> str:
    """如果用户直接给了一段文本描述，包装为标准格式"""
    return text.strip()
