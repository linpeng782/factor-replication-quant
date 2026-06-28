"""训练循环 + early stopping + 结构化记录。

每个 epoch 记录：train/val loss+acc、val rank-IC/ICIR/多空价差、lr、耗时、是否 best。
逐 epoch 追加写 run_dir/history.jsonl，训练后可程序化解析复盘。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from loguru import logger

from .metrics import daily_rank_ic, daily_long_short


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """训练一个 epoch，返回 (loss, accuracy)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += ((pred > 0) == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, np.ndarray]:
    """评估，返回 (loss, accuracy, probs)。probs 按 loader 顺序（shuffle=False）。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    probs = []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logit = model(X_batch)
        loss = criterion(logit, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        correct += ((logit > 0) == y_batch).sum().item()
        total += X_batch.size(0)
        probs.append(torch.sigmoid(logit).cpu().numpy().ravel())

    return total_loss / total, correct / total, np.concatenate(probs)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    val_ret: np.ndarray,
    val_dates: np.ndarray,
    device: str = "cuda",
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 15,
    max_epochs: int = 100,
    run_dir: Path | None = None,
) -> nn.Module:
    """完整训练流程，返回最优模型（按 val_loss 选）。

    val_ret / val_dates: 验证集原始远期收益 + 日期，用于每 epoch 算 rank-IC。
    run_dir: 落 history.jsonl 的目录；None 则只打日志不落盘。
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    hist_path = run_dir / "history.jsonl" if run_dir is not None else None
    if hist_path is not None:
        hist_path.parent.mkdir(parents=True, exist_ok=True)
        hist_path.write_text("")  # 清空

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_probs = evaluate(model, valid_loader, criterion, device)

        # 选股排序类指标（预测概率 vs 真实远期收益）
        ic = daily_rank_ic(val_probs, val_ret, val_dates)
        ls = daily_long_short(val_probs, val_ret, val_dates)
        dt = time.time() - t0

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_ic": round(ic["ic_mean"], 6) if ic["ic_mean"] == ic["ic_mean"] else None,
            "val_icir": round(ic["icir"], 4) if ic["icir"] == ic["icir"] else None,
            "val_ic_pos": round(ic["ic_pos_ratio"], 4) if ic["ic_pos_ratio"] == ic["ic_pos_ratio"] else None,
            "val_long_short": round(ls["long_short"], 6) if ls["long_short"] == ls["long_short"] else None,
            "overfit_gap": round(train_acc - val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": round(dt, 1),
            "is_best": is_best,
        }
        if hist_path is not None:
            with open(hist_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        logger.info(
            f"Epoch {epoch:>3}/{max_epochs} | "
            f"tr loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"IC={ic['ic_mean']:.4f} ICIR={ic['icir']:.2f} L-S={ls['long_short']:.4f} | "
            f"gap={train_acc - val_acc:+.4f} | {dt:.1f}s{' *best' if is_best else ''}"
        )

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (patience={patience}, best epoch={best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)
    logger.success(f"训练完成 | best epoch={best_epoch} val_loss={best_val_loss:.4f}")
    return model
