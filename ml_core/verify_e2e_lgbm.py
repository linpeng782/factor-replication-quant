"""
端到端零漂移验证（LGBM）：同进程、同数据下，ml_core.pipeline.predict_live
与 ml.predict_live 逐元素比对 ŷ（应 max_abs=0），并比信号名单。

为隔离"重构等价"与"磁盘旧面板的数据漂移"，本验证：
  - 在同一进程内现算 ml 参照（用返回值，不依赖磁盘旧面板）；
  - ml.predict_live 会覆盖写 pred_panel_live.parquet → 先备份、验毕还原，不留副作用；
  - 用小窗口（默认近 2 月）即可证明（predict 逐行独立，窗口大小不影响结论）。

运行：ALPHA158_DATA_BACKEND=dquant python -m ml_core.verify_e2e_lgbm
"""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from loguru import logger

from core import config
from ml_core.features import HasFactorPolicy
from ml_core.model import LGBMAdapter
from ml_core.pipeline import PipelineConfig, predict_live
from ml_core.scaling import WholeSetRobustZ
from ml_core.signals import export_panel

RUN_ID = "a158_p27_shap_dquant_htsplit"
WIN_START, WIN_END = "2026-05-01", "2026-06-26"


def main() -> None:
    model_dir = config.ML_MODELS_DIR / RUN_ID
    meta = json.loads((model_dir / "selected_features.json").read_text())
    feats = meta["features"]
    panel_path = config.ML_PREDICTIONS_DIR / RUN_ID / "pred_panel_live.parquet"
    bak_path = panel_path.with_suffix(".parquet.e2e_bak")

    # 备份生产面板（ml.predict_live 会覆盖写）
    had_bak = panel_path.exists()
    if had_bak:
        shutil.copy2(panel_path, bak_path)
    try:
        # ml 参照（同进程现算，用返回值）
        from ml.predict_live import predict_live as ml_predict_live
        old = ml_predict_live(RUN_ID, WIN_START, WIN_END)

        # ml_core 重算
        cfg = PipelineConfig(sources=meta["sources"], neu_sources=meta.get("neu_sources"),
                             has_factor_policy=HasFactorPolicy.NONE, horizon=20, feature_order=feats)
        adapter = LGBMAdapter().load(model_dir)
        scaler = WholeSetRobustZ.load(model_dir / "scaler_x.parquet", feats)
        new = predict_live(model_dir, adapter, scaler, cfg, start=WIN_START, end=WIN_END)

        # ── 比对 ──
        di = old.index.intersection(new.index)
        si = old.columns.intersection(new.columns)
        a = old.loc[di, si].to_numpy(dtype=np.float64)
        b = new.loc[di, si].to_numpy(dtype=np.float64)
        both = np.isfinite(a) & np.isfinite(b)
        max_abs = float(np.nanmax(np.abs(a[both] - b[both]))) if both.any() else 0.0
        only_old = set(old.columns) - set(new.columns)
        only_new = set(new.columns) - set(old.columns)
        logger.info(f"同进程同数据：共有 {len(di)}日×{len(si)}股 | 非空格 {int(both.sum()):,} | "
                    f"ŷ 最大绝对差={max_abs:.3e} | 列差 仅旧={len(only_old)} 仅新={len(only_new)}")
        assert max_abs == 0.0, f"❌ ŷ 不一致 max_abs={max_abs}"
        assert not only_old and not only_new, "❌ 列空间不一致"
        logger.success("  ✅ ml_core.predict_live 与 ml.predict_live 逐元素一致（max_abs=0）")

        # 信号名单一致性（隔离目录，不碰生产信号）
        out_dir = config.ML_ROOT / "_verify_ml_core" / RUN_ID
        export_panel(new, out_dir, top_n=500, rebuild=True)
        old_sig = config.ML_SIGNALS_DIR / RUN_ID
        n_same = n_chk = 0
        for ts in new.index[-5:]:
            fn = f"{ts.strftime('%Y-%m-%d')}.txt"
            of, nf = old_sig / fn, out_dir / fn
            if of.exists() and nf.exists():
                n_chk += 1
                n_same += of.read_text() == nf.read_text()
        logger.success(f"  信号 diff：最近 {n_chk} 份中 {n_same} 份逐行完全一致")
    finally:
        # 还原生产面板
        if had_bak:
            shutil.move(bak_path, panel_path)
            logger.info("  已还原生产 pred_panel_live.parquet")


if __name__ == "__main__":
    main()
