"""
ml_core.model —— 模型适配器（管线唯一真正分叉处）
============================================================
统一接口 ModelAdapter：fit / predict / save / load + (可选) feature_importance。
两条线只在此处分叉，其余管线完全共享。

  LGBMAdapter —— GBDT 回归(MSE)，valid 早停；参数对齐 ml.train.DEFAULT_PARAMS。
  MLPAdapter  —— PyTorch FCNN(158→80→20→1) 二分类(BCEWithLogits)，val_loss 早停；
                 结构/初始化对齐 ml_ht.model.StockMLP + ml_ht.train。

注：参数/结构逐项对齐原实现，保证将来若重训可与历史基线对照（同 seed 同数据可复现，
    MLP 受框架非确定性影响，对齐验证以"加载既有模型只重跑推理"为准，不靠重训比 bit）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from loguru import logger


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def fit(self, X_train, y_train, X_valid, y_valid, **kw) -> "ModelAdapter": ...

    @abstractmethod
    def predict(self, X) -> np.ndarray: ...

    @abstractmethod
    def save(self, model_dir: Path) -> None: ...

    @abstractmethod
    def load(self, model_dir: Path) -> "ModelAdapter": ...


# ==================== LightGBM ====================

LGBM_DEFAULT_PARAMS = dict(
    objective="regression", metric="l2", boosting_type="gbdt",
    learning_rate=0.05, num_leaves=31, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l1=0.0, lambda_l2=0.0, num_threads=64, seed=42,
    deterministic=True, force_row_wise=True, verbose=-1,
)
LGBM_NUM_BOOST_ROUND = 1000
LGBM_EARLY_STOPPING = 200


def _lgbm_live_logger(tag: str, period: int):
    """LightGBM callback：训练中每 period 轮（及第 1 轮）实时把 train/valid loss 写日志
    （对齐 ml.train._make_live_logger，便于边训边看 / 落盘留痕）。"""
    def _cb(env) -> None:
        it = env.iteration + 1                       # env.iteration 0-indexed
        if it != 1 and it % period != 0:
            return
        res = {f"{dn}_{en}": v for dn, en, v, *_ in env.evaluation_result_list}
        logger.info(f"[lgbm-{tag}]   round {it:>4}: "
                    f"train_l2={res.get('train_l2', float('nan')):.6f}  "
                    f"valid_l2={res.get('valid_l2', float('nan')):.6f}")
    _cb.order = 20                                    # 在 early_stopping 之后跑，拿到当轮已算好的结果
    return _cb


class LGBMAdapter(ModelAdapter):
    """GBDT 回归 + valid 早停（对齐 ml.train.train_gbdt）。"""

    name = "lgbm"

    def __init__(self, params: dict | None = None):
        self.params = {**LGBM_DEFAULT_PARAMS, **(params or {})}
        self.booster = None
        self.best_iteration = None

    def fit(self, X_train, y_train, X_valid, y_valid,
            num_boost_round: int = LGBM_NUM_BOOST_ROUND,
            early_stopping_rounds: int = LGBM_EARLY_STOPPING, tag: str = "train",
            log_every: int = 25, **kw):
        import lightgbm as lgb
        dtr = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        dva = lgb.Dataset(X_valid, label=y_valid, reference=dtr, free_raw_data=False)
        hist: dict = {}
        logger.info(f"[lgbm-{tag}] 训练（最多 {num_boost_round} 轮，每 {log_every} 轮记录损失，metric=l2）")
        self.booster = lgb.train(
            self.params, dtr, num_boost_round=num_boost_round,
            valid_sets=[dtr, dva], valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                       lgb.record_evaluation(hist),
                       _lgbm_live_logger(tag, log_every)],
        )
        self.best_iteration = self.booster.best_iteration
        logger.info(f"[lgbm-{tag}] best_iteration={self.best_iteration} "
                    f"valid_l2={hist['valid']['l2'][self.best_iteration-1]:.6f}")
        return self

    def predict(self, X) -> np.ndarray:
        return self.booster.predict(X, num_iteration=self.best_iteration)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        return self.booster.feature_importance(importance_type=importance_type,
                                               iteration=self.best_iteration)

    def save(self, model_dir: Path) -> None:
        model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(model_dir / "model.txt"), num_iteration=self.best_iteration)

    def load(self, model_dir: Path) -> "LGBMAdapter":
        import lightgbm as lgb
        self.booster = lgb.Booster(model_file=str(Path(model_dir) / "model.txt"))
        self.best_iteration = self.booster.best_iteration or None
        return self


# ==================== PyTorch MLP ====================


def _build_mlp(n_features: int = 158, h1: int = 80, h2: int = 20, dropout: float = 0.3):
    """StockMLP：158→80→20→1，Tanh + Dropout + Xavier 初始化（对齐 ml_ht.model.StockMLP）。"""
    import torch.nn as nn
    net = nn.Sequential(
        nn.Linear(n_features, h1), nn.Tanh(), nn.Dropout(dropout),
        nn.Linear(h1, h2), nn.Tanh(), nn.Dropout(dropout),
        nn.Linear(h2, 1),
    )
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class MLPAdapter(ModelAdapter):
    """FCNN 二分类 + val_loss 早停（对齐 ml_ht.train）。"""

    name = "mlp"

    def __init__(self, n_features: int | None = None, device: str | None = None,
                 lr: float = 1e-3, weight_decay: float = 1e-5,
                 patience: int = 15, max_epochs: int = 100, batch_size: int = 8192):
        # n_features 留空即可：fit 时按训练数据列数自动建网、load 时按权重 shape 自适应
        # （单阶段=全特征 / 两阶段=top_k，输入维度由数据决定，无需调用方预先数因子）。
        import torch
        self.n_features = n_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lr, self.weight_decay = lr, weight_decay
        self.patience, self.max_epochs, self.batch_size = patience, max_epochs, batch_size
        self.net = None

    def fit(self, X_train, y_train, X_valid, y_valid, **kw):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.n_features = int(np.asarray(X_train).shape[1])   # 输入维度按训练数据列数定
        self.net = _build_mlp(self.n_features).to(self.device)
        crit = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        def _loader(X, y, shuffle):
            ds = TensorDataset(torch.from_numpy(np.asarray(X, dtype=np.float32)),
                               torch.from_numpy(np.asarray(y, dtype=np.float32)).unsqueeze(1))
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                              drop_last=shuffle, num_workers=0, pin_memory=True)

        tr, va = _loader(X_train, y_train, True), _loader(X_valid, y_valid, False)
        best_loss, best_state, no_improve = float("inf"), None, 0
        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            for xb, yb in tr:
                xb, yb = xb.to(self.device), yb.to(self.device)
                loss = crit(self.net(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            # valid
            self.net.eval(); vloss, n = 0.0, 0
            with torch.no_grad():
                for xb, yb in va:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    l = crit(self.net(xb), yb)
                    vloss += l.item() * xb.size(0); n += xb.size(0)
            vloss /= n
            if vloss < best_loss:
                best_loss, no_improve = vloss, 0
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                no_improve += 1
            logger.info(f"[mlp] epoch {epoch:>3} val_loss={vloss:.4f}{' *best' if no_improve==0 else ''}")
            if no_improve >= self.patience:
                logger.info(f"[mlp] early stop @ {epoch}")
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        import torch
        self.net.eval()
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32))
        out = np.empty(len(Xt), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(Xt), self.batch_size):
                out[i:i + self.batch_size] = torch.sigmoid(
                    self.net(Xt[i:i + self.batch_size].to(self.device))).cpu().numpy().ravel()
        return out

    def save(self, model_dir: Path) -> None:
        import torch
        model_dir = Path(model_dir); model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), model_dir / "model.pt")

    def load(self, model_dir: Path, filename: str | None = None) -> "MLPAdapter":
        """加载权重，兼容两种命名来源（关键：消除 train/serve 串味）：
          - ml_core 自存：model.pt，state_dict 键为 '0.weight'…（裸 Sequential）
          - ml_ht 既有：stock_mlp.pt，键带 'net.' 前缀（StockMLP 把网络包在 self.net 里）
        若检测到统一 'net.' 前缀则剥除，两种命名都能装进同一裸 Sequential（结构逐层对齐）。
        filename 留空时优先 model.pt，回退 stock_mlp.pt。
        """
        import torch
        model_dir = Path(model_dir)
        if filename is not None:
            path = model_dir / filename
        else:
            path = model_dir / "model.pt"
            if not path.exists() and (model_dir / "stock_mlp.pt").exists():
                path = model_dir / "stock_mlp.pt"
        sd = torch.load(path, map_location=self.device)
        if sd and all(k.startswith("net.") for k in sd):   # ml_ht StockMLP 命名 → 剥前缀
            sd = {k[len("net."):]: v for k, v in sd.items()}
        self.n_features = int(sd["0.weight"].shape[1])      # 按首层权重 shape 自适应输入维度
        self.net = _build_mlp(self.n_features)
        self.net.load_state_dict(sd)
        self.net = self.net.to(self.device)
        return self
