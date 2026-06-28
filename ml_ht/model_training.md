# ml_ht 模型训练 —— 复现评估与待办

> 复现对象：华泰证券《人工智能选股之全连接神经网络》（2017-11-23）
> 研报原文：`research-paper-reproduction/parsed_output/华泰证券/20171123-.../full.md`
> 代码目录：`ml_ht/`
> 本文汇总当前实现相对研报方法论的偏离与训练配置问题，供团队对齐。

---

## 一、shuffle / batch_size 配置问题（优先处理）

### 1.1 现状

`dataset.py:build_loaders`（当前版本）：

```python
def build_loaders(data: dict) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        X = torch.from_numpy(data[f"X_{split}"])
        y = torch.from_numpy(data[f"y_{split}"]).unsqueeze(1)
        ds = TensorDataset(X, y)
        n_days = len(np.unique(data[f"{split}_dates"]))
        batch_size = max(X.shape[0] // max(n_days, 1), 1)   # 日均样本数
        loaders[split] = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loaders
```

意图：每天一个 batch（截面 zscore 按天计算，同一天天然是一个截面），训练集不 shuffle。

### 1.2 问题一："按天 batch"是假的

`batch_size = 总样本 / 日数` 算出的是**日均样本数**（train ≈ 3.8M / ~1925 天 ≈ 1976），是一个**固定常数**。但每日股票数是变化的（train 日均 1,348→2,478，AGENTS.md §10 实测）。

固定 batch_size + `shuffle=False` 的真实行为（长表按日期排序，见 `build_long_table.py:227` `np.repeat(dates, N)`）

```
batch 1 = 前 1976 行 = 第 1 天全部(1348) + 第 2 天前 628 行
batch 2 = 第 2 天剩下的(820) + 第 3 天前 1156 行
...
```

即**batch 边界不对齐日期**，每个 batch 是跨好几天的混合切片。docstring 里"每天一个 batch，同一天天然是一个截面"没有做到。

要真正按天切，必须用 `BatchSampler` 按 date 分组（每日行数不等，DataLoader 的固定 `batch_size` 做不到）：

```python
from torch.utils.data import Sampler, DataLoader, TensorDataset
import numpy as np

class DayBatchSampler(Sampler):
    """每天一个 batch；train 时 shuffle 天的顺序，valid/test 保持日期顺序。"""
    def __init__(self, dates: np.ndarray, shuffle: bool):
        self.shuffle = shuffle
        unique = np.unique(dates, return_index=True)
        self.day_batches = []
        idx = np.arange(len(dates))
        for d in np.unique(dates):
            self.day_batches.append(idx[dates == d].tolist())

    def __iter__(self):
        order = np.arange(len(self.day_batches))
        if self.shuffle:
            np.random.shuffle(order)
        for i in order:
            yield self.day_batches[i]

    def __len__(self):
        return len(self.day_batches)

# 用法
sampler = DayBatchSampler(data["train_dates"], shuffle=True)
loader = DataLoader(ds, batch_sampler=sampler)
```

### 1.3 问题二：shuffle=False 在这个场景是有害的

即使把按天 batch 修对了（用上面的 `DayBatchSampler`），`shuffle=False` 仍然有害：

- 长表按日期排序，`shuffle=False` 下**每个 epoch 都从 2010 顺着喂到 2017**，batch 顺序完全一样。
- Adam 带动量，每个 epoch 结束时"最近见到的"永远是 2017 → 权重持续偏向最近市场风格（recency bias）；同时固定 batch 顺序消除了 SGD 该有的随机性，更容易卡在差的局部解。
- 正确做法是 shuffle **"天的先后顺序"**（哪一天先喂随机化），而不是不 shuffle。`DayBatchSampler(shuffle=True)` 即可，天内样本保持在一起。

### 1.4 问题三（更根本）：对这个模型，按天 batch 本身没有意义

`StockMLP`（`model.py`）是**逐样本（pointwise）MLP**：

- 没有 BatchNorm（已 grep 确认，全仓库 0 处）
- forward 里没有任何跨股票的截面操作（没有 ranking 层、没有 over-stocks 的 softmax）
- `BCELoss` 是各样本独立求和

所以**一个 batch 装"某一天"还是"随机混合的多天"，对梯度期望方向没有任何区别**，只影响梯度噪声。截面信息已经在 `preprocess` 的按天 zscore 里用过了，模型本身不需要"一个 batch = 一个截面"。

按天 batch 只有在这些情况下才有价值：
- listwise / 排序损失（batch 内算 IC、rank loss）
- 有 BatchNorm 层（BN 统计量跨样本）
- forward 里做截面运算（如对 batch 内股票做 softmax 归一）

**当前一个都没有**，所以"按天 batch"这一改纯属引入 bug + recency bias，零收益。

### 1.5 建议方案（二选一）

**方案 A（推荐，简单正确）**：回到固定 batch + shuffle
```python
def build_loaders(data: dict, batch_size: int = 8192) -> dict:
    loaders = {}
    for split in ("train", "valid", "test"):
        X = torch.from_numpy(data[f"X_{split}"])
        y = torch.from_numpy(data[f"y_{split}"]).unsqueeze(1)
        ds = TensorDataset(X, y)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )
    return loaders
```

**方案 B（仅当将来要上 IC/排序损失时才值得）**：写真正的 `DayBatchSampler`，训练时 shuffle 天的顺序、天内样本保持在一起。上面 1.2 给了代码骨架。

### 1.6 关于 `drop_last`

当前模型没有 BatchNorm，`drop_last=True` 的唯一作用是防止最后一个 batch 太小时梯度被尾部样本主导。没有 BN 的情况下 `drop_last` 影响微乎其微，设 False 也行。代码注释里写"避免最后一个小 batch 影响 BN"是**错的**（模型里根本没有 BN）。

---

## 二、标签应改为超额收益（研报方法论对齐）

### 现状

`dataset.py:binarize_labels` 用 `forward_return_20d`（绝对收益）按当天截面中位数二分类。

### 研报要求

研报 full.md:381：

> "计算下一整个自然月的个股超额收益（以沪深 300 指数为基准），前 30% 的股票标记为'上涨股'，后 30% 的股票标记为'下跌股'，中间股票标记为'中性股'"

即研报用的是**相对沪深 300 的超额收益**，且是**三分类**（涨/平/跌，前 30% / 中 40% / 后 30%）。

### 影响

alpha158 含大量动量/波动/Beta 类因子，用绝对收益做标签，模型会优先学到**市场 Beta / 波动率溢酬**，而非"选股 alpha"。这正是研报用超额收益的原因。

### 建议

- v1 简化版：`label = forward_return_20d - 当日截面等权均值`（cross-sectional demean，去掉市场暴露）
- v2 严谨版：`label = forward_return_20d - 沪深300 同期收益`（需要加载 benchmark 收益序列）
- 是否上三分类（涨/平/跌）与研报完全对齐，需讨论；二分类简化版可作为对照变体

---

## 三、缺失研报预处理步骤：中位数去极值 + 行业市值中性化

### 研报要求（full.md:383-391）

研报四步预处理：
1. 中位数去极值（MAD，±5×MAD 截断）
2. 缺失值填行业均值
3. 行业市值中性化（对行业哑变量 + log 市值回归取残差）
4. 标准化（zscore）

### 现状

`dataset.py:preprocess` 只做了第 4 步（按天 zscore）。第 2 步被 `has_factor=all` 的严格过滤替代（AGENTS.md §10 已记录这是妥协）；**第 1 步（MAD 去极值）和第 3 步（行业市值中性化）完全缺失**。

### 影响

alpha158 含 EP/BP/市值类因子，不做行业市值中性化会让模型靠"市值效应"白嫖收益；与"标签用绝对收益"叠加（见第二节）形成双重 Beta 暴露。

### 建议

- 至少加截面 MAD 去极值（`alpha_shared/cleaning/` 有现成算法，复用即可）
- 行业市值中性化需要中信行业面板 + 总市值面板（`config.INDUSTRY_PANEL_ZX_PATH` / `MARKET_CAP_PANEL_PATH`，AGENTS.md §1 已列），`alpha_shared/neu/` 有现成中性化算法
- 注意 alpha158 是技术因子，部分（如已对市值中性化的）再做一次中性化可能过度，需个案判断

---

## 四、单次训练 vs 滚动交叉验证

### 研报要求（full.md:393-397, 图表14）

研报分 7 个阶段滚动回测，每段独立训练 + 交叉验证调参。

### 现状

`dataset.py` 的 `SPLIT` 是**一次切分**：train 2010-2017 → valid 2018-2019 → test 2020-2025，test 5 年用同一个模型预测。

### 影响

A 股 2020-2025 风格剧烈切换（抱团瓦解、量化挤兑、微盘股行情），单一 2010-2017 训出的模型在 2022 后大概率显著衰减。

### 建议

- v1：至少在 valid loss 之外给出**逐年 test accuracy 曲线**，看是否单调下滑
- v2：walk-forward（每年滚动重训），与研报对齐

---

## 五、评价闭环不全

### 研报要求

研报核心评价指标：分层回测（5/10 层）+ 信息比率 + F1-score（full.md:355, 443-461）。

### 现状

`train.py` 只打 `loss + accuracy`，`export_signal.py` 直接吐 txt 给外部回测系统，**没有内部验证就交付**。

### 建议

加 test 集 `P(Y=1)` 做**截面 5/10 分层等权组合**，算：
- 多空年化收益、单调性
- IC、ICIR
- F1-score（三分类时）

`alpha_shared/evaluation/` 和 `core/eval_plots.py` 有现成算法可拼。

---

## 六、模型节点数偏离研报经验

### 研报要求（full.md:417）

> "隐层节点选取按经验选取，一般设为输入层节点数的 75%"

研报 70 因子 → 隐层 40、10（约 57% / 14%，论文实际取值与"75%"经验有一段距离，但研报自己给了 75% 这个口径）。

### 现状

`model.py`：`158 → 80 → 20 → 1`（80/158≈0.51，20/158≈0.13）。

### 建议

不算错，但既然是复现研报，按 75% 比例取 `158 → 118 → 30 → 1` 更贴合。可作为 A/B 变体对照（按 AGENTS.md §4 命名 `stock_mlp__h118`）。

---

## 七、路径硬编码 vs config 不一致

- `build_long_table.py:42` 硬编码 `Path("/nfs/ofs-prediction/peterzhenglinpeng/ml/ht")`
- `dataset.py:11` 用 `config.ML_ROOT / "ht"`

两边指向同一处但写法分裂，远端换机或本机化（`FACTOR_REPL_DATA_ROOT` 重定向）会断。

### 建议

`build_long_table.py` 统一用 `config.ML_ROOT / "ht"`。

---

## 八、长表未存 label，dataset 重复 IO

### 现状

`build_long_table.py:187` 读 `LABEL_PATH` 只用于算 `has_label` 布尔，**没把 label 值写进长表**；`dataset.py:44` 又要再读一次 `forward_return_20d.parquet` 并 `.stack().reindex()`。

### 影响

8G 长表 + 2G 标签宽表重复加载，浪费一倍 IO 和一次 stack。

### 建议

build 阶段直接把 `label_20d` 列写进长表（float32，每行一个值，不增加文件体积）。

---

## 九、死代码

- `dataset.py:19` `EMBARGO` 常量未引用（train mask 用 `<=2017-11-30` + valid `>=2018-01-01` 已把 12 月丢掉，embargo 生效，但常量没用上，删掉或注释说明）
- `dataset.py:69` `cross_sectional_zscore` 函数未被调用（`preprocess` 内联了自己的版本），可删

---

## 十、做得对的点（确认不动）

- **can_buy 查 T+1 mask、涨停不过滤** —— 与研报"不剔除涨停"一致，设计干净
- **embargo 21 天 > 20d horizon** —— 防泄漏严谨
- **BCELoss + Sigmoid + Adam(lr=1e-3, weight_decay=1e-5) + early stopping** —— 比研报 TensorFlow 时代写法现代，合理
- **has_factor=all** —— alpha158 MLP 要完整输入，严格过滤是正确取舍
- **NaN≤5 截面中位数填充**（AGENTS.md §10 TODO）—— 作为 v2 优化合理

---

## 优先级汇总

| 优先级 | 待办 | 理由 |
|--------|------|------|
| P0 | 第一节：修复 shuffle/batch 配置 | 当前版本引入 recency bias 且零收益，最紧急 |
| P1 | 第二节：标签超额收益化 | 方法论偏离，影响模型学到的是 Beta 还是 alpha |
| P1 | 第三节：加 MAD 去极值 + 行业市值中性化 | 方法论偏离，与第二节叠加形成双重 Beta 暴露 |
| P2 | 第五节：加分层回测内评 | 不验证就交付等于黑盒出货 |
| P3 | 第四节：滚动重训 | 工程量较大，可作为 v2 |
| P3 | 第六节：模型节点 75% 比例 | A/B 变体对照 |
| P4 | 第七节：路径统一 + 第八节：长表存 label + 第九节：删死代码 | 工程清理 |