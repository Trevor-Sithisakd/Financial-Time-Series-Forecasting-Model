import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import yfinance as yf
import mlflow

from config import (
    TICKERS, START_DATE, END_DATE,
    FORWARD_DAYS, MIN_TRAIN_YRS, MLFLOW_EXPERIMENT,
)
from features import build_features, add_cross_sectional_features
from validation import walk_forward
from evaluation import compute_metrics
from reporting import print_feature_importance, print_quant_metrics

# ── Model selection ───────────────────────────────────────────────────────────
# Swap this import to change which model runs.
# Options: models.linear_baseline | models.xgboost_model
import models.xgboost_model as active_model

CACHE_DIR   = "./cache"
PRICES_FILE = os.path.join(CACHE_DIR, f"prices_{START_DATE}_{END_DATE}.csv")
SPY_FILE    = os.path.join(CACHE_DIR, f"spy_{START_DATE}_{END_DATE}.csv")
MACRO_FILE  = os.path.join(CACHE_DIR, f"macro_{START_DATE}_{END_DATE}.csv")


def load_prices() -> pd.DataFrame:
    if os.path.exists(PRICES_FILE):
        raw = pd.read_csv(PRICES_FILE, header=[0, 1], index_col=0, parse_dates=True)
        cached_tickers = set(raw.columns.get_level_values(1).unique())
        missing = [t for t in TICKERS if t not in cached_tickers]
        if not missing:
            print(f"Loading cached prices from {PRICES_FILE}...")
            return raw
        print(f"Cache missing {len(missing)} tickers {missing} — re-downloading all...")

    print(f"Downloading OHLCV data for {len(TICKERS)} tickers ({START_DATE} to {END_DATE})...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw.to_csv(PRICES_FILE)
    print(f"Cached to {PRICES_FILE}")
    return raw


def load_spy() -> pd.Series:
    if os.path.exists(SPY_FILE):
        series = pd.read_csv(SPY_FILE, index_col=0, parse_dates=True)
        return series.iloc[:, 0].sort_index()
    print("Downloading SPY benchmark data...")
    spy_raw = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    close = spy_raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("SPY").sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_csv(SPY_FILE, header=True)
    return close


def _extract_close(raw, ticker_symbol: str) -> pd.Series:
    """Helper to pull a single close series from a yfinance multi or single download."""
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][ticker_symbol]
    else:
        close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(ticker_symbol)


def load_macro_data(spy_close: pd.Series) -> pd.DataFrame:
    """
    Build a date-indexed macro feature DataFrame from VIX and Treasury yields.
    Cached after first download.

    Features:
      vix_level          : raw VIX — absolute fear level
      vix_5d_chg         : 5-day % change in VIX — direction of fear
      spy_ret_5d         : SPY 5-day return — near-term market trend
      spy_ret_20d        : SPY 20-day return — medium-term market trend
      spy_vol_20d        : SPY 20-day realised vol — market volatility regime
      yield_curve        : 10yr Treasury yield minus 3mo yield (slope)
      yield_curve_20d_chg: 20-day change in yield curve — steepening vs flattening
    """
    if os.path.exists(MACRO_FILE):
        df = pd.read_csv(MACRO_FILE, index_col=0, parse_dates=True)
        print(f"Loading cached macro data from {MACRO_FILE}...")
        return df.sort_index()

    print("Downloading macro data (VIX, 10yr yield, 3mo yield)...")
    vix_raw = yf.download("^VIX", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    tnx_raw = yf.download("^TNX", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)
    irx_raw = yf.download("^IRX", start=START_DATE, end=END_DATE, auto_adjust=False, progress=False)

    vix = _extract_close(vix_raw, "^VIX").rename("vix")
    tnx = _extract_close(tnx_raw, "^TNX").rename("tnx")
    irx = _extract_close(irx_raw, "^IRX").rename("irx")

    idx   = spy_close.index
    vix   = vix.reindex(idx, method="ffill")
    tnx   = tnx.reindex(idx, method="ffill")
    irx   = irx.reindex(idx, method="ffill")
    spy_r = spy_close.pct_change()

    macro = pd.DataFrame(index=idx)
    macro["vix_level"]           = vix
    macro["vix_5d_chg"]          = vix.pct_change(5)
    macro["spy_ret_5d"]          = spy_r.rolling(5).sum()
    macro["spy_ret_20d"]         = spy_r.rolling(20).sum()
    macro["spy_vol_20d"]         = spy_r.rolling(20).std()
    macro["yield_curve"]         = tnx - irx
    macro["yield_curve_20d_chg"] = macro["yield_curve"].diff(20)

    os.makedirs(CACHE_DIR, exist_ok=True)
    macro.to_csv(MACRO_FILE)
    print(f"Cached to {MACRO_FILE}")
    return macro.sort_index()


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    try:
        ed_raw = yf.Ticker(ticker).earnings_dates
        idx    = ed_raw.index
        return pd.DatetimeIndex(idx.tz_localize(None) if idx.tz is not None else idx)
    except Exception:
        return pd.DatetimeIndex([])


def fetch_earnings_surprise(ticker: str) -> pd.Series:
    try:
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return pd.Series(dtype=float, name="earnings_surprise")
        if "EPS Estimate" not in ed.columns or "Reported EPS" not in ed.columns:
            return pd.Series(dtype=float, name="earnings_surprise")
        ed = ed.dropna(subset=["EPS Estimate", "Reported EPS"])
        if ed.empty:
            return pd.Series(dtype=float, name="earnings_surprise")
        surprise = (ed["Reported EPS"] - ed["EPS Estimate"]) / ed["EPS Estimate"].abs()
        idx = ed.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)
        surprise.index = idx
        return surprise.sort_index().rename("earnings_surprise")
    except Exception:
        return pd.Series(dtype=float, name="earnings_surprise")


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{active_model.MODEL_NAME} + macro + cross-sectional"):
        mlflow.log_params({
            "model"         : active_model.MODEL_NAME,
            "tickers"       : ",".join(TICKERS),
            "start_date"    : START_DATE,
            "end_date"      : END_DATE,
            "forward_days"  : FORWARD_DAYS,
            "min_train_yrs" : MIN_TRAIN_YRS,
            "target"        : "excess_return_over_SPY",
            "new_features"  : "macro(vix,yields,spy_momentum),cross_sectional_ranks,peer_excess_ret",
        })

        raw        = load_prices()
        spy_close  = load_spy()
        macro_data = load_macro_data(spy_close)

        # ── Phase 1: build base features for every ticker ─────────────────────
        all_feat_dfs     = {}
        all_earn_dates   = {}
        all_earn_surp    = {}

        for ticker in TICKERS:
            prices            = raw.xs(ticker, level=1, axis=1)[["Close", "High", "Low", "Volume"]].dropna()
            earn_dates        = fetch_earnings_dates(ticker)
            earn_surp         = fetch_earnings_surprise(ticker)
            all_earn_dates[ticker] = earn_dates
            all_earn_surp[ticker]  = earn_surp

            feat_df = build_features(
                prices, earn_dates, FORWARD_DAYS,
                benchmark_close=spy_close,
                earnings_surprise=earn_surp,
                macro_data=macro_data,
            )
            all_feat_dfs[ticker] = feat_df
            print(f"  {ticker}: {feat_df.shape[0]} rows x {feat_df.shape[1]} columns (pre cross-sectional)")

        # ── Phase 2: add cross-sectional ranks across all tickers ─────────────
        all_feat_dfs = add_cross_sectional_features(all_feat_dfs)
        print(f"\n  Cross-sectional features added. "
              f"Final feature count: {all_feat_dfs[TICKERS[0]].shape[1]} columns\n")

        # ── Phase 3: stack all tickers and run ONE cross-sectional model ─────────
        # Each row is one ticker-date pair. The model trains on all companies
        # simultaneously, learning generalisable patterns rather than per-ticker noise.
        stacked = pd.concat([
            df.assign(ticker=ticker)
            for ticker, df in all_feat_dfs.items()
        ]).sort_index()

        print(f"  Stacked dataset: {stacked.shape[0]:,} rows x {stacked.shape[1]} columns "
              f"({len(all_feat_dfs)} tickers x ~{stacked.shape[0] // len(all_feat_dfs)} rows each)\n")

        results, pred_df, model, feature_cols = walk_forward(
            stacked, MIN_TRAIN_YRS, active_model.build
        )

        if results.empty:
            print("\nNo results — check data availability.")
            return

        # Walk-forward summary per year (now covers all tickers combined)
        print("Walk-forward results (all tickers combined):")
        print(results[["test_year", "n_train", "n_test", "mae", "directional_accuracy"]]
              .to_string(index=False))

        # Feature importances
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            print_feature_importance({"cross_sectional": pd.Series(
                importances, index=feature_cols
            ).sort_values(ascending=False)})

        # Per-ticker IC summary from the single model's predictions
        print(f"\n{'-'*50}")
        print("  PER-TICKER METRICS (single cross-sectional model)")
        print(f"{'-'*50}")
        for ticker in TICKERS:
            t_preds = pred_df[pred_df["ticker"] == ticker] if "ticker" in pred_df.columns else pd.DataFrame()
            if t_preds.empty:
                continue
            tm = compute_metrics(t_preds, FORWARD_DAYS)
            print(f"  {ticker:<8}  IC: {tm['ic']:+.4f}  "
                  f"Sharpe: {tm['sharpe']:+.3f}  "
                  f"Dir Acc: {tm['directional_accuracy']:.3f}")

        # Aggregate quant metrics across all tickers
        metrics = compute_metrics(pred_df, FORWARD_DAYS)
        print_quant_metrics(active_model.MODEL_NAME, metrics)

        mlflow.log_metrics({
            "ic"                  : metrics["ic"],
            "rank_ic"             : metrics["rank_ic"],
            "directional_accuracy": metrics["directional_accuracy"],
            "top_decile_return"   : metrics["top_decile_return"],
            "sharpe"              : metrics["sharpe"],
            "mean_mae"            : results["mae"].mean(),
        })

        for ticker in TICKERS:
            t_preds = pred_df[pred_df["ticker"] == ticker] if "ticker" in pred_df.columns else pd.DataFrame()
            if t_preds.empty:
                continue
            tm = compute_metrics(t_preds, FORWARD_DAYS)
            mlflow.log_metrics({f"{ticker}_ic"    : tm["ic"],
                                 f"{ticker}_sharpe": tm["sharpe"]})

        print("\nDone.")
        print(f"MLflow run logged under experiment: '{MLFLOW_EXPERIMENT}'")
        print("View results: mlflow ui")


if __name__ == "__main__":
    main()
