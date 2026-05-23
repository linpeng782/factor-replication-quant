"""
一次性迁移脚本：把 flat specs/ inputs/ docs/ 重组为 sources/<publisher>/<group>/。

用法：
    python scripts/migrate_to_sources.py --dry-run     # 只打印计划，不动文件
    python scripts/migrate_to_sources.py               # 实跑

迁移完成后这个脚本就完成历史使命，但保留在仓库作为审计痕迹。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SPECS_DIR = ROOT / "specs"
INPUTS_DIR = ROOT / "inputs"
DOCS_DIR = ROOT / "docs"
SOURCES_DIR = ROOT / "sources"


# ─────────────────────────── 映射表 ──────────────────────────────────
# (相对源路径, 相对目标路径) — 都相对于 ROOT
# 目标目录会自动 mkdir -p。

KYSEC_27 = "sources/kysec/paper_27_microstructure"
FOUNDER_01 = "sources/founder/paper_01_moderate_risk"

NPF = "sources/fundamental/npf_series"
ROE = "sources/fundamental/roe_series"
ROIC = "sources/fundamental/roic_series"
PE = "sources/fundamental/pe_series"
REG = "sources/fundamental/cross_section_regress"

KYSEC_27_SPECS = [
    "eruption_followup_ratio",
    "eruption_turnover_corr",
    "eruption_turnover_sensitivity",
    "peak_interval_kurt",
    "peak_interval_skew",
    "peak_interval_std",
    "peak_minute_count",
    "peak_ridge_price_ratio",
    "peak_ridge_turnover_ratio",
    "peak_weighted_quantile",
    "peakridge_minute_corr",
    "ridge_interval_kurt",
    "ridge_interval_skew",
    "ridge_interval_std",
    "ridge_minute_count",
    "ridge_minute_return",
    "ridge_relative_vwap",
    "valley_relative_vwap",
    "valley_ridge_price_ratio",
    "valley_weighted_quantile",
]
KYSEC_27_DOCS = [
    "eruption_turnover_corr.md",
    "peak_interval_kurt.md",
    "peak_minute_count.md",
]

NPF_FACTORS = [
    "npf_apoq_mrq",
    "npf_ayoy_mrq",
    "npf_mrq_accs8",
    "npf_mrq_sue8",
    "npf_pqoq_mrq",
    "npf_pyoy_mrq",
]
NPF_DOCS = ["npf_mrq_accs8.md", "npf_mrq_sue8.md"]

ROE_FACTORS = [
    "roe_apoq_mrq",
    "roe_ayoy_mrq",
    "roe_mrq_new",
    "roe_pqoq_mrq",
    "roe_pyoy_mrq",
]

ROIC_FACTORS = [
    "roic_ttm_all_rnk8",
    "roic_ttm_dev_std8",
    "roic_ttm_ind_rnk8",
]
ROIC_INPUTS = [f + ".md" for f in ROIC_FACTORS]
ROIC_DOCS = [
    "roic_ttm_all_rnk8.md",
    "roic_ttm_dev_std8.md",
    "roic_ttm_ind_rnk8.md",
    "roic_ttm_factors_research.md",
    "roic_ttm_issues.md",
]

PE_FACTORS = ["pe_mrq", "pe_ttm_delta60", "pe_ttm_new"]

REG_FACTORS = ["reg_pb_gshe", "reg_pe_hist"]
REG_INPUTS = ["reg_pb_gshe.md", "reg_pe_hist.md"]
REG_DOCS = ["reg_pb_gshe.md", "reg_pe_hist_progress.md"]


def _build_moves() -> list[tuple[str, str]]:
    """返回 [(rel_src, rel_dst), ...]"""
    moves: list[tuple[str, str]] = []

    # ─ kysec/paper_27_microstructure
    moves.append(("inputs/开源_微观_27.md", f"{KYSEC_27}/input.md"))
    for f in KYSEC_27_SPECS:
        moves.append((f"specs/{f}", f"{KYSEC_27}/specs/{f}"))
    for d in KYSEC_27_DOCS:
        moves.append((f"docs/{d}", f"{KYSEC_27}/docs/{d}"))

    # ─ founder/paper_01_moderate_risk
    moves.append(("inputs/适度冒险因子.md", f"{FOUNDER_01}/input.md"))
    moves.append(("specs/moderate_risk", f"{FOUNDER_01}/specs/moderate_risk"))
    # publisher-level README（评估文档归到 founder/ 顶层）
    moves.append(
        ("docs/founder_securities_8_papers_evaluation.md", "sources/founder/README.md")
    )

    # ─ fundamental/npf_series
    for f in NPF_FACTORS:
        moves.append((f"inputs/{f}.md", f"{NPF}/inputs/{f}.md"))
        moves.append((f"specs/{f}", f"{NPF}/specs/{f}"))
    for d in NPF_DOCS:
        moves.append((f"docs/{d}", f"{NPF}/docs/{d}"))

    # ─ fundamental/roe_series（无 inputs/docs）
    for f in ROE_FACTORS:
        moves.append((f"specs/{f}", f"{ROE}/specs/{f}"))

    # ─ fundamental/roic_series
    for f in ROIC_FACTORS:
        moves.append((f"inputs/{f}.md", f"{ROIC}/inputs/{f}.md"))
        moves.append((f"specs/{f}", f"{ROIC}/specs/{f}"))
    for d in ROIC_DOCS:
        moves.append((f"docs/{d}", f"{ROIC}/docs/{d}"))

    # ─ fundamental/pe_series（无 inputs/docs）
    for f in PE_FACTORS:
        moves.append((f"specs/{f}", f"{PE}/specs/{f}"))

    # ─ fundamental/cross_section_regress
    for f in REG_FACTORS:
        moves.append((f"specs/{f}", f"{REG}/specs/{f}"))
    for f in REG_INPUTS:
        moves.append((f"inputs/{f}", f"{REG}/inputs/{f}"))
    for d in REG_DOCS:
        moves.append((f"docs/{d}", f"{REG}/docs/{d}"))

    return moves


# 迁移完成后顶层留下来的文件 / 目录（白名单，用来检测漏迁）
EXPECTED_TOP_RESIDUE = {
    "docs": {"multi_factor_paper_architecture.md"},
    "inputs": set(),
    "specs": set(),
}


def _check_no_leftovers(strict: bool) -> list[str]:
    """检查迁移后 specs/ inputs/ docs/ 是否还有未迁的文件。strict=True 时 raise。"""
    issues: list[str] = []
    for top, expected in EXPECTED_TOP_RESIDUE.items():
        d = ROOT / top
        if not d.exists():
            continue
        actual = {p.name for p in d.iterdir()}
        leftover = actual - expected
        if leftover:
            issues.append(f"{top}/ 还有未迁文件：{sorted(leftover)}")
    if issues and strict:
        for i in issues:
            print(f"⚠ {i}")
        raise SystemExit("迁移有遗漏，请检查映射表")
    return issues


def _do_move(src_rel: str, dst_rel: str, dry_run: bool) -> str:
    src = ROOT / src_rel
    dst = ROOT / dst_rel
    if not src.exists():
        return f"  ✗ MISSING  {src_rel}"
    if dst.exists():
        return f"  ✗ DST EXISTS {dst_rel}（拒绝覆盖）"
    if dry_run:
        return f"  → {src_rel}  →  {dst_rel}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return f"  ✓ {src_rel}  →  {dst_rel}"


def _create_placeholder_readmes(dry_run: bool) -> None:
    """各 paper / series 创建占位 README（如果还没有）。"""
    placeholders = {
        f"{KYSEC_27}/README.md": (
            "# 开源证券·市场微观结构系列（27）\n\n"
            "**研报**：高频成交量的峰、岭、谷信息（2025-07-20，魏建榕/王志豪）\n"
            "**共享方法论**：日内分钟成交量按「过去 20 日同时点 1σ」划分喷发/温和；"
            "孤立喷发=量峰，连续喷发=量岭，温和=量谷。\n\n"
            "因子清单见 specs/ 子目录。详见 input.md（研报原文）+ docs/<factor>.md（沉淀）。\n"
        ),
        f"{FOUNDER_01}/README.md": (
            "# 方正证券·适度冒险因子（成交量激增时刻 alpha）\n\n"
            "**研报**：2022-04-12 曹春晓\n"
            "**共享方法论**：日内成交量激增时刻识别 + 耀眼 5 分钟波动率/收益率 → 适度冒险因子\n\n"
            "详见 input.md。\n"
        ),
        f"{NPF}/README.md": (
            "# 单季度净利润（NPF）系列\n\n"
            "共享方法论：基于米筐 `net_profit_mrq_n`（已是单季度，禁 diff）做 SUE / "
            "环比 / 同比 / 季节性过滤等变体。详见各 docs/<factor>.md。\n"
        ),
        f"{ROE}/README.md": "# ROE 系列（单季度变体）\n",
        f"{ROIC}/README.md": "# ROIC_TTM 系列（稳定性 / 行业排名 / 全市场排名）\n",
        f"{PE}/README.md": "# PE 系列（mrq / ttm 变体）\n",
        f"{REG}/README.md": "# 截面回归类因子（PB-ROE 残差等）\n",
        "sources/internal/.gitkeep": "",
    }
    for rel, content in placeholders.items():
        path = ROOT / rel
        if path.exists():
            continue
        if dry_run:
            print(f"  + (dry) create {rel}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"  + created {rel}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="只打印不动文件")
    parser.add_argument(
        "--strict-leftover",
        action="store_true",
        help="迁移后若 specs/inputs/docs 还有未预期残留，报错退出",
    )
    args = parser.parse_args()

    moves = _build_moves()
    print(f"=== 迁移计划：{len(moves)} 项 ===")

    missing = []
    dst_conflict = []
    for src_rel, dst_rel in moves:
        result = _do_move(src_rel, dst_rel, dry_run=args.dry_run)
        print(result)
        if "MISSING" in result:
            missing.append(src_rel)
        elif "DST EXISTS" in result:
            dst_conflict.append(dst_rel)

    if missing:
        print(f"\n⚠ {len(missing)} 个源文件缺失：")
        for m in missing:
            print(f"  - {m}")
    if dst_conflict:
        print(f"\n⚠ {len(dst_conflict)} 个目标已存在（拒绝覆盖）：")
        for d in dst_conflict:
            print(f"  - {d}")

    print("\n=== 占位 README / .gitkeep ===")
    _create_placeholder_readmes(dry_run=args.dry_run)

    if not args.dry_run:
        print("\n=== 检查残留 ===")
        issues = _check_no_leftovers(strict=args.strict_leftover)
        if issues:
            for i in issues:
                print(f"  ⚠ {i}")
        else:
            print("  ✓ 顶层 specs/ inputs/ docs/ 已清空（除架构文档）")

    if missing or dst_conflict:
        sys.exit(1)


if __name__ == "__main__":
    main()
