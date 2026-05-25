# 因子相关性分析 — microstructure_41

- 因子数：**41**
- 区间：cleaned panel 全交集（按 inventory direction 校正）
- 聚类：scipy.linkage(method='average')，threshold = 1-ρ < 0.4 (ρ > 0.6 同簇)

## 1. 簇划分（12 个簇）

- **簇 1（4 个，簇内 avg ρ=+0.838）**：
  - `pj_peak_interval_kurt`
  - `pj_peak_interval_skew`
  - `pj_peak_interval_std`
  - `pj_peak_minute_count`
- **簇 2（4 个，簇内 avg ρ=+0.787）**：
  - `peak_interval_kurt`
  - `peak_interval_skew`
  - `peak_interval_std`
  - `peak_minute_count`
- **簇 3（独立）**：`peakridge_minute_corr`
- **簇 4（3 个，簇内 avg ρ=+0.838）**：
  - `peak_weighted_quantile`
  - `pj_valley_weighted_quantile`
  - `valley_weighted_quantile`
- **簇 5（2 个，簇内 avg ρ=+0.748）**：
  - `pj_ridge_minute_return`
  - `ridge_minute_return`
- **簇 6（3 个，簇内 avg ρ=+0.828）**：
  - `pj_valley_relative_vwap`
  - `pj_valley_ridge_price_ratio`
  - `pj_valley_ridge_price_ratio__mp10`
- **簇 7（7 个，簇内 avg ρ=+0.850）**：
  - `valley_relative_vwap`
  - `peak_ridge_price_ratio`
  - `peak_ridge_price_ratio__mp10`
  - `ridge_relative_vwap`
  - `ridge_relative_vwap__mp10`
  - `valley_ridge_price_ratio`
  - `valley_ridge_price_ratio__mp10`
- **簇 8（4 个，簇内 avg ρ=+0.888）**：
  - `eruption_turnover_sensitivity`
  - `pj_jump_turnover_sensitivity`
  - `eruption_turnover_corr`
  - `pj_jump_turnover_corr`
- **簇 9（7 个，簇内 avg ρ=+0.769）**：
  - `pj_jump_followup_ratio`
  - `eruption_followup_ratio`
  - `peak_ridge_turnover_ratio`
  - `ridge_minute_count`
  - `ridge_interval_std`
  - `ridge_interval_kurt`
  - `ridge_interval_skew`
- **簇 10（4 个，簇内 avg ρ=+0.729）**：
  - `pj_ridge_interval_kurt`
  - `pj_ridge_interval_skew`
  - `pj_peak_ridge_turnover_ratio`
  - `pj_ridge_minute_count`
- **簇 11（独立）**：`pj_ridge_interval_std`
- **簇 12（独立）**：`pj_peakridge_minute_corr`

## 2. 最独立因子排行（avg |ρ| 升序）

| 排名 | 因子 | avg \|ρ\| |
|-----|------|------------|
| 1 | `peakridge_minute_corr` | 0.070 |
| 2 | `peak_weighted_quantile` | 0.128 |
| 3 | `pj_valley_weighted_quantile` | 0.159 |
| 4 | `pj_peak_interval_kurt` | 0.167 |
| 5 | `pj_peakridge_minute_corr` | 0.169 |
| 6 | `pj_peak_interval_std` | 0.170 |
| 7 | `pj_peak_interval_skew` | 0.174 |
| 8 | `peak_interval_kurt` | 0.176 |
| 9 | `pj_peak_minute_count` | 0.187 |
| 10 | `peak_interval_skew` | 0.191 |

## 3. 冗余对 Top-10（|ρ| 降序）

| 因子 A | 因子 B | ρ |
|--------|--------|----|
| `peak_ridge_price_ratio` | `peak_ridge_price_ratio__mp10` | +0.993 |
| `ridge_interval_kurt` | `ridge_interval_skew` | +0.990 |
| `ridge_relative_vwap` | `ridge_relative_vwap__mp10` | +0.987 |
| `pj_peak_interval_kurt` | `pj_peak_interval_skew` | +0.986 |
| `valley_ridge_price_ratio` | `valley_ridge_price_ratio__mp10` | +0.985 |
| `pj_valley_ridge_price_ratio` | `pj_valley_ridge_price_ratio__mp10` | +0.983 |
| `peak_interval_kurt` | `peak_interval_skew` | +0.980 |
| `pj_ridge_interval_kurt` | `pj_ridge_interval_skew` | +0.979 |
| `pj_valley_weighted_quantile` | `valley_weighted_quantile` | +0.936 |
| `eruption_turnover_sensitivity` | `pj_jump_turnover_sensitivity` | +0.928 |

## 4. 推荐去重（41 → 12）

**保留**（每簇取簇内最独立的代表）：
- `pj_peak_interval_kurt`
- `peak_interval_kurt`
- `peakridge_minute_corr`
- `peak_weighted_quantile`
- `ridge_minute_return`
- `pj_valley_ridge_price_ratio__mp10`
- `ridge_relative_vwap__mp10`
- `eruption_turnover_sensitivity`
- `ridge_interval_std`
- `pj_ridge_interval_kurt`
- `pj_ridge_interval_std`
- `pj_peakridge_minute_corr`

**淘汰**（簇内冗余）：
- `pj_peak_interval_skew`
- `pj_peak_interval_std`
- `pj_peak_minute_count`
- `peak_interval_skew`
- `peak_interval_std`
- `peak_minute_count`
- `pj_valley_weighted_quantile`
- `valley_weighted_quantile`
- `pj_ridge_minute_return`
- `pj_valley_relative_vwap`
- `pj_valley_ridge_price_ratio`
- `valley_relative_vwap`
- `peak_ridge_price_ratio`
- `peak_ridge_price_ratio__mp10`
- `ridge_relative_vwap`
- `valley_ridge_price_ratio`
- `valley_ridge_price_ratio__mp10`
- `pj_jump_turnover_sensitivity`
- `eruption_turnover_corr`
- `pj_jump_turnover_corr`
- `pj_jump_followup_ratio`
- `eruption_followup_ratio`
- `peak_ridge_turnover_ratio`
- `ridge_minute_count`
- `ridge_interval_kurt`
- `ridge_interval_skew`
- `pj_ridge_interval_skew`
- `pj_peak_ridge_turnover_ratio`
- `pj_ridge_minute_count`
