"""
Spec 生成层：研报文字描述 → spec.yaml（单段式 LLM + 校验重试闭环）

调用 LLM（Moonshot Kimi 兼容 OpenAI SDK）一次产出 yaml；用 spec_schema.validate_spec
做静态校验，校验失败时把错误回传给 LLM 自我纠正，最多重试 N 次。

公开入口：
    generate_spec_from_research(input_text: str, factor_name: str) -> dict

会写出：
    specs/<factor_name>/spec.yaml          LLM 直产并校验通过的 yaml
    specs/<factor_name>/.llm_session.json  保留 LLM 完整对话历史（含 thinking 和重试）
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from core.spec_schema import SpecError, validate_spec


PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
SPECS_DIR = Path(__file__).parent.parent / "specs"
PROMPT_FILE = PROMPTS_DIR / "research_to_yaml.md"

DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 1.0   # kimi-k2.6 仅接受 1.0；其他模型可在调用时显式覆盖


# ── 提取 yaml 块 ────────────────────────────────────────────


_YAML_BLOCK_RE = re.compile(r"<spec_yaml>\s*(.*?)\s*</spec_yaml>", re.DOTALL)
_THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.DOTALL)
_FENCE_YAML_RE = re.compile(r"```ya?ml\s*(.*?)\s*```", re.DOTALL)


def _extract_yaml(text: str) -> str:
    """从 LLM 输出抠出 yaml 块。优先 <spec_yaml>，否则尝试 ```yaml 围栏，否则整段。"""
    m = _YAML_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FENCE_YAML_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_thinking(text: str) -> Optional[str]:
    m = _THINKING_RE.search(text)
    return m.group(1).strip() if m else None


# ── LLM 客户端 ────────────────────────────────────────────


def _load_env():
    """加载 .env（不引入额外依赖，自己 parse 一下）"""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _make_client():
    _load_env()
    from openai import OpenAI

    api_key = os.environ.get("MOONSHOT_API_KEY")
    base_url = os.environ.get("MOONSHOT_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "缺少 MOONSHOT_API_KEY 或 MOONSHOT_BASE_URL；请检查 .env"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _model_name() -> str:
    return os.environ.get("MOONSHOT_MODEL", "moonshot-v1-32k")


# ── 主流程 ────────────────────────────────────────────────


def generate_spec_from_research(
    input_text: str,
    factor_name: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict:
    """
    研报文字 → spec.yaml（含 LLM 自我纠错循环）

    成功时把 yaml 写到 specs/<factor_name>/spec.yaml 并返回 spec dict。
    失败时 raise RuntimeError。
    """
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"prompt 文件不存在: {PROMPT_FILE}")
    system_prompt = PROMPT_FILE.read_text(encoding="utf-8")

    client = _make_client()
    model = _model_name()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"# 输入研报\n{input_text.strip()}\n\n# 期望 factor.name\n`{factor_name}`",
        },
    ]
    session = []  # 完整对话日志，最后落盘

    spec: Optional[dict] = None
    last_err: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        logger.info(f"[spec_gen] LLM 调用 attempt {attempt}/{max_retries} (model={model})")
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature
        )
        text = resp.choices[0].message.content or ""
        thinking = _extract_thinking(text)
        yaml_text = _extract_yaml(text)

        session.append(
            {
                "attempt": attempt,
                "thinking": thinking,
                "yaml_text": yaml_text,
                "raw": text,
            }
        )
        if thinking:
            logger.info(f"[spec_gen] thinking:\n{thinking}")

        # 解析 + 校验
        try:
            spec_candidate = yaml.safe_load(yaml_text)
            if not isinstance(spec_candidate, dict):
                raise yaml.YAMLError("LLM 输出顶层不是 dict")
            validate_spec(spec_candidate)
        except (yaml.YAMLError, SpecError, KeyError) as e:
            last_err = f"{type(e).__name__}: {e}"
            logger.warning(f"[spec_gen] attempt {attempt} 失败: {last_err}")
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "上次输出违反约束：\n"
                        f"{last_err}\n\n"
                        "请保持完整 <thinking> + <spec_yaml> 输出协议，"
                        "修复违反点后重新输出整份 yaml。"
                    ),
                }
            )
            continue

        spec = spec_candidate
        logger.info(f"[spec_gen] ✅ attempt {attempt} 校验通过")
        break

    # 写盘
    factor_dir = SPECS_DIR / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)
    session_path = factor_dir / ".llm_session.json"
    session_path.write_text(
        json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if spec is None:
        raise RuntimeError(
            f"spec 生成失败（已重试 {max_retries} 次）；"
            f"最后错误: {last_err}；详见 {session_path}"
        )

    spec_path = factor_dir / "spec.yaml"
    header = (
        "# 由 core/spec_generator 从研报自动产出，已通过 spec_schema 静态校验。\n"
        "# 编辑后请重新跑 validate_spec 确认；完整 LLM 对话见 .llm_session.json。\n"
    )
    spec_path.write_text(
        header + yaml.dump(spec, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.info(f"[spec_gen] ✅ 写入 {spec_path}")
    return spec


def load_spec_yaml(factor_name: str) -> dict:
    """读 specs/<factor>/spec.yaml；运行流水线时用。"""
    spec_path = SPECS_DIR / factor_name / "spec.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec 文件不存在: {spec_path}")
    return yaml.safe_load(spec_path.read_text(encoding="utf-8"))


# ── 独立 CLI ────────────────────────────────────────────────
# 用法（研报文件按约定放在 inputs/<factor_name>.md）：
#     python -m core.spec_generator npf_mrq_sue8


if __name__ == "__main__":
    import argparse
    import sys

    p = argparse.ArgumentParser(
        description="从研报文字生成 spec.yaml（LLM + spec_schema 校验闭环）",
    )
    p.add_argument("factor_name", help="目标 factor 英文名（snake_case）")
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="（可选）覆盖默认输入路径；默认按约定读 inputs/<factor_name>.md",
    )
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    args = p.parse_args()

    INPUTS_DIR = Path(__file__).parent.parent / "inputs"
    input_path = args.input or (INPUTS_DIR / f"{args.factor_name}.md")
    if not input_path.exists():
        p.error(
            f"输入文件不存在: {input_path}\n"
            f"约定路径: inputs/<factor_name>.md，或用 --input 显式指定"
        )

    text = input_path.read_text(encoding="utf-8")
    try:
        spec = generate_spec_from_research(
            text,
            args.factor_name,
            max_retries=args.max_retries,
            temperature=args.temperature,
        )
    except Exception as e:
        logger.error(f"❌ {e}")
        sys.exit(1)
    print(yaml.dump(spec, allow_unicode=True, sort_keys=False))
