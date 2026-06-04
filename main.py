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
from features import build_features
from validation import walk_forward
from evaluation import compute_metrics
from reporting import (
    print_ticker_results, print_aggregate_summary,
    print_feature_importance, print_quant_metrics,
)

# ── Model selection ───────────────────────────────────────────────────────────
# Swap this import to change which model runs.
# Options: models.linear_baseline | models.xgboost_model
import models.xgboost_model as active_model

CACHE_DIR   = "./cache"
PRICES_FILE = os.path.join(CACHE_DIR, f"prices_{START_DATE}_{END_DATE}.csv")
SPY_FILE    = os.path.join(CACHE_DIR, f"spy_{START_DATE}_{END_DATE}.csv")


def load_prices() -> pd.DataFrame:
    if os.path.exists(PRICES_FILE):
        print(f"Loading cached prices from {PRICES_FILE}...")
        return pd.read_csv(PRICES_FILE, header=[0, 1], index_col=0, parse_dates=True)
    print(f"Downloading OHLCV data for {TICKERS} ({START_DATE} to {END_DATE})...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    os.makedirs(CACHE_DIR, exist_ok=True)
    raw.to_csv(PRICES_FILE)
    print(f"Cached to {PRICES_FILE}")
    return raw


def load_spy() -> pd.Series:
    if os.path.exists(SPY_FILE):
        series = pd.read_csv(SPY_FILE, index_col=0, parse_dates=True)
        return series.iloc[:, 0].rename("SPY").sort_index()
    print("Downloading SPY benchmark data...")
    spy_raw = yf.download("SPY", start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    close = spy_raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.rename("SPY").sort_index()
    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_csv(SPY_FILE, header=True)
    return close


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    try:
        ed_raw = yf.Ticker(ticker).earnings_dates
        idx    = ed_raw.index
        return pd.DatetimeIndex(idx.tz_localize(None) if idx.tz is not None else idx)
    except Exception:
        return pd.DatetimeIndex([])


def fetch_earnings_surprise(ticker: str) -> pd.Series:
    """
    Returns a date-indexed Series of normalised earnings surprise values.
    surprise = (Reported EPS - EPS Estimate) / abs(EPS Estimate)
    Only populated where yfinance has both estimate and reported figures.
    """
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
        n = len(surprise)
        print(f"    earnings surprise: {n} quarters of data available")
        return surprise.sort_index().rename("earnings_surprise")
    except Exception:
        return pd.Series(dtype=float, name="earnings_surprise")


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{active_model.MODEL_NAME} + earnings surprise"):
        mlflow.log_params({
            "model"          : active_model.MODEL_NAME,
            "tickers"        : ",".join(TICKERS),
            "start_date"     : START_DATE,
            "end_date"       : END_DATE,
            "forward_days"   : FORWARD_DAYS,
            "min_train_yrs"  : MIN_TRAIN_YRS,
            "target"         : "excess_return_over_SPY",
            "new_features"   : "macd,bollinger,atr_norm,earnings_surprise",
        })

        raw       = load_prices()
        spy_close = load_spy()

        all_results     = []
        all_importances = {}
        all_preds       = []

        for ticker in TICKERS:
            prices = raw.xs(ticker, level=1, axis=1)[
                ["Close", "High", "Low", "Volume"]
            ].dropna()
            earnings_dates   = fetch_earnings_dates(ticker)
            earnings_surprise = fetch_earnings_surprise(ticker)
            feat_df = build_features(
                prices, earnings_dates, FORWARD_DAYS,
                benchmark_close=spy_close,
                earnings_surprise=earnings_surprise,
            )

            results, pred_df, model, feature_cols = walk_forward(
                feat_df, MIN_TRAIN_YRS, active_model.build
            )

            if results.empty:
                print(f"\n  {ticker}: not enough data for walk-forward validation.")
                continue

            results["ticker"] = ticker
            pred_df["ticker"] = ticker
            all_results.append(results)
            all_preds.append(pred_df)

            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                all_importances[ticker] = pd.Series(
                    importances, index=feature_cols
                ).sort_values(ascending=False)

            print_ticker_results(ticker, feat_df.shape, results, pred_df)

        if not all_results:
            print("\nNo results — check data availability.")
            return

        combined       = pd.concat(all_results, ignore_index=True)
        combined_preds = pd.concat(all_preds, ignore_index=True)

        print_aggregate_summary(combined)

        if all_importances:
            print_feature_importance(all_importances)

        metrics = compute_metrics(combined_preds, FORWARD_DAYS)
        print_quant_metrics(active_model.MODEL_NAME, metrics)

        mlflow.log_metrics({
            "ic"                  : metrics["ic"],
            "rank_ic"             : metrics["rank_ic"],
            "directional_accuracy": metrics["directional_accuracy"],
            "top_decile_return"   : metrics["top_decile_return"],
            "sharpe"              : metrics["sharpe"],
            "mean_mae"            : combined["mae"].mean(),
        })

        for ticker in TICKERS:
            ticker_preds = combined_preds[combined_preds["ticker"] == ticker]
            if ticker_preds.empty:
                continue
            t_metrics = compute_metrics(ticker_preds, FORWARD_DAYS)
            mlflow.log_metrics({f"{ticker}_ic"    : t_metrics["ic"],
                                 f"{ticker}_sharpe": t_metrics["sharpe"]})

        print("\nDone.")
        print(f"MLflow run logged under experiment: '{MLFLOW_EXPERIMENT}'")
        print("View results: mlflow ui")


if __name__ == "__main__":
    main()
