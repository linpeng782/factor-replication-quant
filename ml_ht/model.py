"""FCNN 模型定义：158 → 80 → 20 → 1。"""
import torch
import torch.nn as nn


class StockMLP(nn.Module):
    """二分类全连接神经网络。

    结构: Linear(158,80) → Tanh → Dropout → Linear(80,20) → Tanh → Dropout → Linear(20,1)
    输出 logits（不含 Sigmoid），配合 BCEWithLogitsLoss 使用。
    """

    def __init__(self, n_features: int = 158, hidden1: int = 80, hidden2: int = 20, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Tanh 网络用 Xavier/Glorot 初始化（PyTorch 默认 Kaiming 是为 ReLU 设计的）。"""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
