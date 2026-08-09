"""
quota 侧袋信号生成 —— 主模型 + 基本面分域侧袋融合
============================================================
组合构造(信号文件顺序 = 买入优先级):
  前 N 行:  侧袋股票(晚期动量域 × 成长复合分 top N,roe_new 低档否决)
  其余行:  基线模型信号(去重后依序补满)

侧袋规则 v0(源自 experiment_conditional_double_sort.py 验证):
  域:    60日涨幅截面前 10%(晚期动量)
  打分:  成长复合 = mean(sue8, npf_pyoy, roe_pyoy 的域内分位)
  否决:  roe_new 域内分位 < 1/3(崩盘保护)
  入选:  域内复合分前 N 名

输出:
  <ML_PREDICTIONS_DIR>/quota_sp{N}/signals/YYYY-MM-DD.txt   (格式与基线一致: 日期_代码)
  (放 predictions/<run>/signals/ 层级,回测结果目录按父目录名 quota_sp{N} 命名,避免互相覆盖)

用法:改参数区后在仓库根
  PYTHONPATH=. python scripts/build_quota_sidepocket_signals.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

import config

# ════════════════════ 参数区（手动改这里，支持环境变量覆盖） ════════════════════
import os
BASE_RUN = os.environ.get("BASE_RUN", "lgbm_a158_p27_shap_dq_pool2")  # 基线信号来源
SIDEPOCKET_NS = [int(x) for x in os.environ.get("SIDEPOCKET_NS", "20,10").split(",")]
MOM_WINDOW = int(os.environ.get("MOM_WINDOW", "60"))                  # 动量窗口
MOM_TOP_PCT = float(os.environ.get("MOM_TOP_PCT", "0.10"))           # 动量域 = 前 10%
VETO_PCT = float(os.environ.get("VETO_PCT", str(1 / 3)))             # roe_new 域内分位低于该值 → 否决
RUN_TAG = os.environ.get("RUN_TAG", "")                               # 输出目录后缀(如 _w20)
SIG_START = os.environ.get("SIG_START", "2020-01-02")                 # 信号起点(与基线对齐)
SIG_END = os.environ.get("SIG_END", "2026-06-30")
# ══════════════════════════════════════════════════════════════

_CXL = config.RAW_FACTOR_BASE / "cxl-dquant"
SCORE_FACTORS = {
    "sue8": _CXL / "npf_series" / "npf_mrq_sue8.parquet",
    "npf_pyoy": _CXL / "npf_series" / "npf_pyoy_mrq.parquet",
    "roe_pyoy": _CXL / "roe_series" / "roe_pyoy_mrq.parquet",
}
VETO_FACTOR = _CXL / "roe_series" / "roe_mrq_new.parquet"


def main():
    base_sig_dir = config.ML_PREDICTIONS_DIR / BASE_RUN / "signals"
    out_dirs = {n: config.ML_PREDICTIONS_DIR / f"quota_sp{n}{RUN_TAG}" / "signals" for n in SIDEPOCKET_NS}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── 数据加载 ──────────────────────────────────────────
    vwap = pd.read_parquet(config.VWAP_PANEL_PATH)
    vwap.index = pd.to_datetime(vwap.index)
    logger.info(f"vwap 面板 {vwap.shape}")

    panels = {}
    for name, fp in SCORE_FACTORS.items():
        df = pd.read_parquet(fp)
        df.index = pd.to_datetime(df.index)
        panels[name] = df
    veto_panel = pd.read_parquet(VETO_FACTOR)
    veto_panel.index = pd.to_datetime(veto_panel.index)
    logger.info(f"打分因子 {list(panels)} + 否决因子 roe_mrq_new 加载完成")

    # ── 逐信号日生成 ──────────────────────────────────────
    sig_files = sorted(base_sig_dir.glob("*.txt"))
    P = vwap.values
    dates = vwap.index
    cols = vwap.columns
    n_done, n_skip = 0, 0
    stats = []

    for sf in sig_files:
        d_str = sf.stem
        d = pd.Timestamp(d_str)
        if d < pd.Timestamp(SIG_START) or d > pd.Timestamp(SIG_END):
            continue

        base_lines = [line.strip() for line in sf.read_text().splitlines() if line.strip()]

        # 信号日必须在 vwap 面板内且有 60d 历史,否则侧袋为空、原样透传
        pos = dates.searchsorted(d)
        sp_codes: list[str] = []
        if pos < len(dates) and dates[pos] == d and pos >= MOM_WINDOW:
            ret60 = P[pos] / P[pos - MOM_WINDOW] - 1
            valid = np.isfinite(ret60)
            if valid.sum() >= 500:
                cutoff = np.nanquantile(ret60[valid], 1 - MOM_TOP_PCT)
                domain = valid & (ret60 >= cutoff)

                # 域内成长复合分(三因子域内分位均值,要求全部非 NaN)
                fvals = {}
                for name, fpanel in panels.items():
                    r = fpanel.index.searchsorted(d, side="right") - 1
                    fvals[name] = fpanel.iloc[r].reindex(cols).values if r >= 0 else np.full(len(cols), np.nan)
                r = veto_panel.index.searchsorted(d, side="right") - 1
                veto_vals = veto_panel.iloc[r].reindex(cols).values if r >= 0 else np.full(len(cols), np.nan)

                m = domain.copy()
                for v in fvals.values():
                    m &= np.isfinite(v)
                m &= np.isfinite(veto_vals)

                if m.sum() >= 30:
                    idx = np.where(m)[0]
                    pct = np.zeros(len(idx))
                    for v in fvals.values():
                        pct += pd.Series(v[idx]).rank(pct=True).values
                    pct /= len(fvals)
                    veto_pct = pd.Series(veto_vals[idx]).rank(pct=True).values
                    keep = veto_pct >= VETO_PCT
                    order = np.argsort(-pct[keep])
                    sp_codes = list(cols[idx[keep][order]])

        # ── 组装两个 N 版本 ────────────────────────────
        for n, out_dir in out_dirs.items():
            sp_n = sp_codes[:n]
            sp_set = set(sp_n)
            merged = [f"{d_str}_{c}" for c in sp_n]
            for line in base_lines:
                code = line.split("_", 1)[1] if "_" in line else line
                if code not in sp_set:
                    merged.append(line)
                if len(merged) >= 500:
                    break
            (out_dir / sf.name).write_text("\n".join(merged) + "\n")

        stats.append(dict(date=d_str, sp_n=len(sp_codes[:max(SIDEPOCKET_NS)])))
        n_done += 1
        if not sp_codes:
            n_skip += 1

    st = pd.DataFrame(stats)
    logger.info(f"生成完成: {n_done} 个信号日 | 侧袋为空 {n_skip} 天")
    logger.info(f"侧袋平均席位 {st['sp_n'].mean():.1f} | 满席({max(SIDEPOCKET_NS)}只)占比 "
                f"{(st['sp_n'] >= max(SIDEPOCKET_NS)).mean():.0%}")
    for n, d in out_dirs.items():
        logger.info(f"  N={n}: {d}")


if __name__ == "__main__":
    main()
