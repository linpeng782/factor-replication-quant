"""
输出层：复现报告生成

功能：
1. 对比复现结果与 Spec 中的预期效果
2. 数据质量检查（缺失率、异常值）
3. 集成 evaluation 结果（IC / 分层回测 / 单调性）
4. 生成 Markdown 报告

输出：
- output/<factor_name>/report.md
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from core.config import OUTPUT_DIR


def generate_report(
    factor_name: str,
    factor_df: pd.DataFrame,
    spec_yaml: dict,
    evaluation_results: Optional[dict] = None,
) -> Path:
    """
    生成复现报告

    Args:
        factor_name: 因子名称
        factor_df: 计算出的因子值 DataFrame
        spec_yaml: Spec YAML dict
        evaluation_results: 评估结果（如有），来自 evaluate_single_factor 的返回值

    Returns:
        报告文件路径
    """
    factor_dir = OUTPUT_DIR / factor_name
    factor_dir.mkdir(parents=True, exist_ok=True)
    report_path = factor_dir / "report.md"

    lines = []
    lines.append(f"# 因子复现报告：{factor_name}")
    lines.append("")
    lines.append(f"- **中文名**: {spec_yaml['factor'].get('name_cn', 'N/A')}")
    lines.append(f"- **定义**: {spec_yaml['factor'].get('description', 'N/A')}")
    lines.append(f"- **方向**: {'正向' if spec_yaml['factor'].get('direction') == 1 else '反向'}")
    lines.append("")

    # 1. 数据质量检查
    lines.append("## 一、数据质量检查")
    lines.append("")
    quality = _check_data_quality(factor_df)
    lines.append(f"| 检查项 | 结果 |")
    lines.append(f"|--------|------|")
    for item, result in quality.items():
        lines.append(f"| {item} | {result} |")
    lines.append("")

    # 2. 因子值统计
    lines.append("## 二、因子值统计")
    lines.append("")
    numeric_cols = factor_df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats = factor_df[numeric_cols].describe().T
        lines.append(f"| 列名 | 均值 | 标准差 | 最小值 | 中位数 | 最大值 | 缺失率 |")
        lines.append(f"|------|------|--------|--------|--------|--------|--------|")
        for col in numeric_cols:
            s = stats.loc[col]
            missing = factor_df[col].isna().mean()
            lines.append(
                f"| {col} | {s['mean']:.4f} | {s['std']:.4f} | "
                f"{s['min']:.4f} | {s['50%']:.4f} | {s['max']:.4f} | {missing:.2%} |"
            )
    lines.append("")

    # 3. 评估结果（新增）
    lines.append("## 三、因子评估")
    lines.append("")

    if evaluation_results and evaluation_results.get("success"):
        direction = evaluation_results["direction"]
        ic_summary = evaluation_results.get("ic_summary")
        layered_summary = evaluation_results.get("layered_summary")
        monotonicity = evaluation_results.get("monotonicity", np.nan)

        lines.append(f"**Direction**: {direction:+d} {'(原始方向)' if direction == 1 else '(已翻转)'}")
        lines.append("")

        # IC 汇总
        if ic_summary is not None:
            lines.append("### IC 指标（Spearman Rank IC）")
            lines.append("")
            lines.append(f"| 持有期 | IC Mean | IC Std | ICIR | t-stat | Positive% | n_days |")
            lines.append(f"|--------|---------|--------|------|--------|-----------|--------|")
            for h, row in ic_summary.iterrows():
                lines.append(
                    f"| {h} | {row['ic_mean']:+.4f} | {row['ic_std']:.4f} | "
                    f"{row['icir']:+.3f} | {row['ic_t']:+.2f} | "
                    f"{row['pct_positive']:.1%} | {int(row['n_days'])} |"
                )
            lines.append("")

        # 分层回测汇总
        if layered_summary is not None:
            lines.append("### 分层回测（5组 × 5日调仓）")
            lines.append("")
            lines.append(f"| 分组 | 年化收益 | 年化波动 | Sharpe | 平均换手 |")
            lines.append(f"|------|----------|----------|--------|----------|")
            for grp, row in layered_summary.iterrows():
                lines.append(
                    f"| {grp} | {row['ann_return']:+.2%} | {row['ann_vol']:.2%} | "
                    f"{row['sharpe']:+.3f} | {row['mean_turnover']:.1%} |"
                )
            lines.append("")
            lines.append(f"**单调性**（Pearson 分组序 vs 年化收益）: {monotonicity:+.3f}")
            lines.append("")

        # 与预期对比
        lines.append("### 与预期效果对比")
        lines.append("")
        validation = spec_yaml.get("validation", {})
        lines.append(f"| 指标 | 预期范围 | 实际值 | 是否达标 |")
        lines.append(f"|------|----------|--------|----------|")

        if ic_summary is not None and f"{spec_yaml.get('evaluation', {}).get('primary_horizon', 5)}d" in ic_summary.index:
            h = f"{spec_yaml.get('evaluation', {}).get('primary_horizon', 5)}d"
            ic_mean = ic_summary.loc[h, "ic_mean"]
            ic_ir = ic_summary.loc[h, "icir"]
            expected_ic = validation.get("expected_ic_mean", [0, 0])
            expected_ir = validation.get("expected_ic_ir", [0, 0])

            ic_ok = expected_ic[0] <= abs(ic_mean) <= expected_ic[1] if not np.isnan(ic_mean) else False
            ir_ok = expected_ir[0] <= abs(ic_ir) <= expected_ir[1] if not np.isnan(ic_ir) else False

            lines.append(
                f"| IC均值 | [{expected_ic[0]}, {expected_ic[1]}] | {ic_mean:.4f} | {'✅' if ic_ok else '❌'} |"
            )
            lines.append(
                f"| IC_IR | [{expected_ir[0]}, {expected_ir[1]}] | {ic_ir:.4f} | {'✅' if ir_ok else '❌'} |"
            )
        else:
            lines.append("| IC均值 | 见预期 | 评估完成 | - |")
            lines.append("| IC_IR | 见预期 | 评估完成 | - |")
        lines.append("")
    else:
        lines.append("评估未执行或失败。")
        if evaluation_results and evaluation_results.get("error"):
            lines.append(f"错误信息: {evaluation_results['error']}")
        lines.append("")

    # 4. 与原研报的差异
    lines.append("## 四、与原研报的差异记录")
    lines.append("")
    lines.append("| 研报原文 | Spec理解 | 偏差说明 |")
    lines.append("|----------|----------|----------|")
    formula = spec_yaml.get("formula", {})
    lines.append(f"| 因子公式 | {formula.get('expression', 'N/A')} | 按Spec执行 |")
    lines.append("")

    # 5. 风险提示
    lines.append("## 五、风险提示")
    lines.append("")
    warnings = spec_yaml.get("risk_warnings", [])
    if warnings:
        for w in warnings:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w["level"], "⚪")
            lines.append(f"- {emoji} **[{w['level']}]** {w['item']}: {w['description']}")
    else:
        lines.append("- 无特别风险提示")
    lines.append("")

    # 6. 输出文件清单
    lines.append("## 六、输出文件")
    lines.append("")
    lines.append(f"- `{factor_name}.parquet` — 原始因子值面板")
    if evaluation_results and evaluation_results.get("success"):
        lines.append(f"- `{factor_name}_cleaned.parquet` — 清洗后因子值面板")
        lines.append(f"- `evaluation.png` — 2×2 综合评估图")
        lines.append(f"- `ic_summary.csv` — IC 指标汇总")
        lines.append(f"- `layered_summary.csv` — 分层回测汇总")
    lines.append("")

    # 7. 结论
    lines.append("## 七、结论")
    lines.append("")
    if evaluation_results and evaluation_results.get("success"):
        lines.append("因子评估已完成，结果见上文。")
    else:
        lines.append("因子值面板已生成，评估部分执行失败或未启用。")
    lines.append("")
    lines.append(f"---")
    lines.append(f"*报告生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"📊 复现报告已生成: {report_path}")
    return report_path


def _check_data_quality(df: pd.DataFrame) -> Dict[str, str]:
    """数据质量检查"""
    results = {}

    # 总行数
    results["总行数"] = str(len(df))

    # 缺失率检查
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        max_missing = df[numeric_cols].isna().mean().max()
        results["最大缺失率"] = f"{max_missing:.2%}"
        if max_missing > 0.5:
            results["缺失率评估"] = "🔴 过高，请检查"
        elif max_missing > 0.2:
            results["缺失率评估"] = "🟡 偏高，注意"
        else:
            results["缺失率评估"] = "✅ 正常"

    # 异常值检查（3倍标准差外）
    outlier_count = 0
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            outliers = df[(df[col] - mean).abs() > 3 * std][col]
            outlier_count += len(outliers)
    results["极端异常值数量(>3σ)"] = str(outlier_count)

    # 重复检查
    dup = df.duplicated().sum()
    results["重复行数"] = f"{dup} ({dup/len(df):.2%})" if len(df) > 0 else "0"

    return results
