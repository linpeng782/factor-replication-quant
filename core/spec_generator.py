"""
Spec 生成层：将输入文本/PDF → Spec Markdown → Spec YAML

支持两种模式：
1. manual 模式：生成 Prompt 文件，用户手动贴到 LLM（如 kimi CLI），将结果保存回来
2. auto 模式：自动调用 LLM API 生成（需要配置 API Key）

输出：
- specs/<factor_name>/spec.md      人类可读的 Spec 文档
- specs/<factor_name>/spec.yaml    机器可读的 YAML 配置
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple

import yaml


PROMPT_DIR = Path(__file__).parent.parent / "prompts"
SPECS_DIR = Path(__file__).parent.parent / "specs"


def generate_spec(
    input_text: str,
    factor_name: Optional[str] = None,
    mode: str = "manual",
    llm_config: Optional[dict] = None,
) -> Tuple[Path, Path]:
    """
    主入口：输入文本 → 生成 Spec Markdown + Spec YAML

    Args:
        input_text: 因子描述文本（研报原文或用户输入）
        factor_name: 因子英文名称，如未提供则尝试从文本中提取
        mode: "manual" 或 "auto"
        llm_config: auto 模式下的 LLM 配置 {provider, api_key, model, base_url}

    Returns:
        (spec_md_path, spec_yaml_path)
    """
    # 1. 确定因子名
    if factor_name is None:
        factor_name = _extract_factor_name(input_text)
    factor_name = factor_name.strip().lower()

    factor_dir = SPECS_DIR / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)

    spec_md_path = factor_dir / "spec.md"
    spec_yaml_path = factor_dir / "spec.yaml"

    # 2. 生成 Spec Markdown
    if mode == "manual":
        spec_md = _generate_spec_manual(input_text, factor_name, spec_md_path)
    else:
        spec_md = _generate_spec_auto(input_text, factor_name, llm_config or {})
        spec_md_path.write_text(spec_md, encoding="utf-8")

    # 3. 生成 Spec YAML
    if mode == "manual":
        # manual 模式下，YAML 也由用户手动提供，或后续单独提取
        _generate_yaml_manual(spec_md, factor_name, spec_yaml_path)
    else:
        spec_yaml = _extract_yaml_from_spec_auto(spec_md, llm_config or {})
        spec_yaml_path.write_text(spec_yaml, encoding="utf-8")

    print(f"✅ Spec 已生成: {factor_dir}")
    print(f"   - Markdown: {spec_md_path}")
    print(f"   - YAML: {spec_yaml_path}")

    if mode == "manual":
        print("\n⚠️  当前为 manual 模式，请按以下步骤操作：")
        print(f"   1. 打开 {factor_dir}/prompt_for_llm.txt，复制内容到 kimi CLI")
        print("   2. 将 LLM 输出的 Spec Markdown 保存到 spec.md")
        print("   3. 再复制 spec.md 内容到 kimi CLI，要求生成 YAML")
        print("   4. 将 YAML 保存到 spec.yaml")

    return spec_md_path, spec_yaml_path


def _extract_factor_name(text: str) -> str:
    """从文本中提取因子名称（第一行冒号/中文冒号前的英文单词）"""
    lines = text.strip().splitlines()
    for line in lines:
        line = line.strip()
        # 支持英文冒号 : 和中文冒号 ：
        for sep in [":", "："]:
            if sep in line:
                name = line.split(sep)[0].strip()
                if name and re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", name):
                    return name
    return "unknown_factor"


def _generate_spec_manual(input_text: str, factor_name: str, spec_md_path: Path) -> str:
    """
    Manual 模式：生成 Prompt 文件供用户手动贴到 LLM
    不直接生成 spec.md，而是生成 prompt_for_llm.txt
    """
    factor_dir = spec_md_path.parent
    factor_dir.mkdir(parents=True, exist_ok=True)

    # 读取 Prompt 模板
    prompt_template_path = PROMPT_DIR / "spec_generation.txt"
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{input_content}", input_text)

    prompt_path = factor_dir / "prompt_for_llm.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    # 同时生成一个空的 spec.md 占位
    placeholder = (
        f"# Spec 文档：{factor_name}\n\n"
        f"> ⚠️ 当前为 manual 模式。请将 prompt_for_llm.txt 的内容发送给 LLM，\n"
        f"> 然后将 LLM 的回复保存到此文件，覆盖本占位内容。\n"
    )
    spec_md_path.write_text(placeholder, encoding="utf-8")

    return placeholder


def _generate_yaml_manual(spec_md: str, factor_name: str, spec_yaml_path: Path) -> None:
    """
    Manual 模式：生成 YAML 提取的 Prompt 文件
    同时放一个占位 YAML
    """
    factor_dir = spec_yaml_path.parent
    prompt_template_path = PROMPT_DIR / "yaml_extraction.txt"
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt = prompt_template.replace("{spec_md}", spec_md)

    prompt_path = factor_dir / "prompt_for_yaml.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    placeholder = (
        f"# Spec YAML: {factor_name}\n"
        f"# ⚠️ 当前为 manual 模式。请将 prompt_for_yaml.txt 的内容发送给 LLM，\n"
        f"# 然后将生成的 YAML 保存到此文件，覆盖本占位内容。\n"
    )
    spec_yaml_path.write_text(placeholder, encoding="utf-8")


def _generate_spec_auto(input_text: str, factor_name: str, llm_config: dict) -> str:
    """
    Auto 模式：自动调用 LLM API 生成 Spec Markdown
    目前支持 OpenAI 兼容接口（Kimi / Zhipu 等）
    """
    prompt_template_path = PROMPT_DIR / "spec_generation.txt"
    prompt = prompt_template_path.read_text(encoding="utf-8").replace(
        "{input_content}", input_text
    )

    provider = llm_config.get("provider", "openai")
    api_key = llm_config.get("api_key", os.environ.get("MOONSHOT_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    model = llm_config.get("model", os.environ.get("MOONSHOT_MODEL", "kimi-k2.6"))
    base_url = llm_config.get("base_url", os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"))

    if not api_key:
        raise ValueError(
            "auto 模式需要提供 API Key，请设置 llm_config['api_key'] 或环境变量 OPENAI_API_KEY"
        )

    try:
        import openai
    except ImportError:
        raise ImportError("auto 模式需要安装 openai: pip install openai")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一名量化研究专家。"},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=4096,
    )

    return response.choices[0].message.content


def _extract_yaml_from_spec_auto(spec_md: str, llm_config: dict) -> str:
    """
    Auto 模式：自动调用 LLM API 从 Spec Markdown 提取 YAML
    """
    prompt_template_path = PROMPT_DIR / "yaml_extraction.txt"
    prompt = prompt_template_path.read_text(encoding="utf-8").replace(
        "{spec_md}", spec_md
    )

    provider = llm_config.get("provider", "openai")
    api_key = llm_config.get("api_key", os.environ.get("MOONSHOT_API_KEY", os.environ.get("OPENAI_API_KEY", "")))
    model = llm_config.get("model", os.environ.get("MOONSHOT_MODEL", "kimi-k2.6"))
    base_url = llm_config.get("base_url", os.environ.get("MOONSHOT_BASE_URL", "https://api.moonshot.cn/v1"))

    try:
        import openai
    except ImportError:
        raise ImportError("auto 模式需要安装 openai: pip install openai")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个结构化数据提取专家，只输出 YAML，不输出其他内容。"},
            {"role": "user", "content": prompt},
        ],
        temperature=1,
        max_tokens=4096,
    )

    content = response.choices[0].message.content
    # 尝试提取代码块中的 YAML
    match = re.search(r"```yaml\n(.*?)\n```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content.strip()


# ── 便捷函数 ────────────────────────────────

def load_spec_yaml(factor_name: str) -> dict:
    """加载已生成的 Spec YAML"""
    path = SPECS_DIR / factor_name / "spec.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Spec YAML 不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_spec_md(factor_name: str) -> str:
    """加载已生成的 Spec Markdown"""
    path = SPECS_DIR / factor_name / "spec.md"
    if not path.exists():
        raise FileNotFoundError(f"Spec Markdown 不存在: {path}")
    return path.read_text(encoding="utf-8")
