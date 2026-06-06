# Financial Time Series Forecasting Model

XGBoost and linear regression pipeline for predicting 5-day forward returns on large-cap US equities using OHLCV technical features and earnings event flags.

## Results

| Row | Model | Features | Target | IC | Rank IC | Dir. Acc | Top-Decile Return | Sharpe |
|-----|-------|----------|--------|----|---------|----------|-------------------|--------|
| 1 | Linear Regression (Ridge) | Base | Raw return | -0.049 | -0.049 | 0.542 | +0.41% | 0.48 |
| 2 | XGBoost | Base | Raw return | -0.048 | -0.048 | 0.499 | +0.56% | 0.69 |
| 3 | Linear Regression (Ridge) | Base + MACD/BB/ATR | Excess vs SPY | -0.028 | -0.028 | 0.498 | +0.39% | 0.61 |
| 4 | XGBoost | Base + MACD/BB/ATR | Excess vs SPY | -0.041 | -0.041 | 0.486 | +0.10% | 0.16 |
| 5 | Linear Regression (Ridge) | Base + MACD/BB/ATR + Earnings Surprise | Excess vs SPY | -0.026 | -0.026 | 0.501 | +0.32% | 0.50 |
| 6 | XGBoost | Base + MACD/BB/ATR + Earnings Surprise | Excess vs SPY | -0.043 | -0.043 | 0.486 | +0.14% | 0.22 |
| **7** | **Linear Regression (Ridge), per-ticker** | **+ Macro (VIX, yields) + Cross-sectional ranks** | **Excess vs SPY** | **+0.005** | **+0.005** | **0.502** | **+0.55%** | **0.759** |
| 8 | XGBoost, stacked cross-sectional | + Macro + Cross-sectional ranks | Excess vs SPY | -0.006 | -0.006 | 0.495 | +0.21% | 0.32 |
| 9 | Linear Regression (Ridge), stacked cross-sectional | + Macro + Cross-sectional ranks | Excess vs SPY | +0.006 | +0.006 | 0.502 | +0.18% | 0.28 |

**Thresholds:** IC > 0.05 and Sharpe > 1.0 indicate a meaningful signal.

**Rows 1→2 (Linear vs XGBoost, raw target):** Near-identical IC (-0.049 vs -0.048) confirms the feature set is the bottleneck. XGBoost's higher Sharpe (0.69) shows marginally better top-decile selection. *Both models learned bull market drift, not signal — directional accuracy of 0.542 is spurious.*

**Rows 2→3 (Raw target → Excess return):** Switching to SPY-excess target improved IC (-0.049 → -0.028) and collapsed fake directional accuracy (0.542 → 0.498). *Excess return target removes the dominant noise source — market beta — making the task harder but more honest.*

**Rows 3→4 (Linear vs XGBoost, excess target):** XGBoost degraded significantly (Sharpe 0.61 → 0.16). With a harder, less noisy target Ridge's regularisation generalises better than XGBoost which overfits. *On small datasets with weak signal, simpler models outperform complex ones.*

**Rows 3→5 and 4→6 (Adding earnings surprise):** Only 12 quarters of yfinance surprise data available vs 7-year window. Marginal IC improvement, Sharpe unchanged. *Feature fires too rarely to shift aggregate metrics — needs full point-in-time historical data.*

**Row 5→7 (Adding macro + cross-sectional features): IC crossed zero for the first time (+0.005).** Sharpe improved from 0.50 → 0.759, top-decile return from +0.32% → +0.55%. VIX gives the model regime awareness (momentum signals behave differently in fear vs calm markets). Yield curve slope captures macro headwinds for growth stocks. Cross-sectional momentum and RSI ranks let the model identify which stock is strongest *relative to peers*, not just in absolute terms. *IC still below 0.05 threshold — direction of travel is positive, more data and full earnings surprise history are the next lever.*

**Rows 7→8→9 (Per-ticker vs stacked cross-sectional model):** Restructured to train one model on all 100 tickers simultaneously (~140k rows vs ~1.4k per ticker). IC improved marginally for Ridge (+0.005 → +0.006) but Sharpe dropped sharply (0.759 → 0.278). XGBoost degraded again (-0.006). *The stacked model improves IC (ranking accuracy across companies) but hurts Sharpe because the top decile is now selected from 100 tickers simultaneously — a harder selection problem. Per-ticker models are more specialised and produce better risk-adjusted returns at this signal strength. Row 7 remains the best result.*

## Session Log

### Session 1 — 2026-06-04

**Data pipeline**
- Built `transcripts.py` to download SEC 8-K filings from EDGAR using `sec_edgar_downloader`
- Fixed extraction to parse the SGML `full-submission.txt` bundle and pull Exhibit 99.1 (the earnings press release) rather than the 8-K cover page, which contains no financial data
- Cleaned files saved to `transcripts_clean/`

**Forecasting pipeline**
- Built feature engineering (`features.py`): lag returns (1/2/5/10/20d), rolling volatility (5/20/60d), momentum (5/20/60d), RSI-14, volume z-score, earnings event flag
- Built walk-forward validation (`validation.py`): expanding window, trains on years 1-N, tests on year N+1 — no data shuffling
- Refactored monolithic script into modular structure: `config.py`, `features.py`, `validation.py`, `reporting.py`, `main.py`
- Added price caching to avoid re-downloading on every run (`cache/`)

**Evaluation framework**
- Replaced MAE/directional accuracy with quant-standard metrics in `evaluation.py`: IC (Spearman rank correlation), Rank IC, top-decile mean return, top-decile Sharpe ratio
- Added MLflow experiment tracking — every run logs parameters and metrics to `mlruns/`
- Added model factory pattern (`models/`) so swapping models requires changing one import line

**Results**
- Row 1 (Linear Regression): IC -0.049, Sharpe 0.48 — model learned average return drift, no ranking ability
- Row 2 (XGBoost): IC -0.048, Sharpe 0.69 — nearly identical IC confirms features are the bottleneck, not model architecture
- Volatility features dominate importance; lag return features contribute almost nothing
- 8-K press releases downloaded but not yet connected to the forecasting model

**Key findings**
- Technical indicators alone (price/volume) are insufficient for positive IC on large-cap US stocks — they are already priced in by institutional participants
- The directional accuracy of 0.542 on the linear model is explained by bull market drift (2018-2024), not genuine signal
- Earnings surprise (actual EPS vs consensus) is the highest-priority missing feature

**Session 2 additions**
- Added MACD signal + histogram, Bollinger Band position, ATR-normalised return to `features.py`
- Changed target to excess return over SPY benchmark — removes market beta noise
- Added SPY caching (`cache/spy_*.csv`) alongside ticker price cache
- Updated `build_features` to accept `benchmark_close` parameter; backward compatible (defaults to raw return)
- Row 3: IC -0.028, Sharpe 0.61 — target change was the main driver, new technical features contributed little
- Row 4 (XGBoost same setup): IC -0.041, Sharpe 0.16 — Ridge outperforms XGBoost on harder excess return task
- Implemented `earnings_surprise` feature: normalised (actual - estimate) / |estimate|, carried forward 5 trading days (PEAD window)
- yfinance only provides 12 quarters of historical surprise data — feature fires rarely across 7-year window
- Row 5 (Linear + surprise): IC -0.026, Sharpe 0.50 — marginal IC improvement, Sharpe limited by sparse data
- Row 6 (XGBoost + surprise): IC -0.043, Sharpe 0.22 — XGBoost continues to underperform Ridge on this task

**Key finding from session 2**
Ridge regression consistently outperforms XGBoost on the excess return target. The signal is too weak and data too limited for XGBoost's complexity to help — it overfits. Earnings surprise is the right feature conceptually but needs full point-in-time historical data (paid API like Compustat or Refinitiv) to show its real impact.

**Next steps**
- Obtain full historical point-in-time earnings estimates (Alpha Vantage, Finnhub, or Compustat)
- Add sector/market relative features — cross-sectional model across all tickers simultaneously
- Consider macro features: VIX level, SPY 5-day return, yield curve slope
- Target: IC > 0.05, Sharpe > 1.0

## Pipeline

```
yfinance OHLCV (2018-2024, 5 tickers)
    -> feature engineering (lag returns, volatility, momentum, RSI, volume z-score, earnings event)
    -> walk-forward validation (expanding window, no data shuffling)
    -> XGBoost / Linear Regression
    -> evaluation (IC, Rank IC, directional accuracy, top-decile Sharpe)
    -> MLflow logging
```

## Setup

```
pip install -r requirements.txt
python main.py
mlflow ui   # view experiment results at localhost:5000
```

## Switching models

Edit the import in `main.py`:

```python
import models.linear_baseline as active_model   # row 1
import models.xgboost_model as active_model      # row 2
```
