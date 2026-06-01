"""
Financial Time Series Forecasting Pipeline
Walk-forward XGBoost on OHLCV + technical features + earnings event flag.

Target: 5-day forward return per ticker.
Validation: expanding-window walk-forward (no data shuffling).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# -- Config --------------------------------------------------------------------

TICKERS       = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
FORWARD_DAYS  = 5   # predict return over this many trading days
MIN_TRAIN_YRS = 3   # minimum years of history before first test window

# -- Feature engineering -------------------------------------------------------

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(prices: pd.DataFrame, earnings_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build a feature matrix from a single-ticker OHLCV DataFrame.
    All features are lagged so they only use information available at time t.
    Target is computed from future prices and dropped from X at train time.
    """
    close  = prices["Close"]
    volume = prices["Volume"]
    ret    = close.pct_change()

    feat = pd.DataFrame(index=prices.index)

    # Lag returns — information available at close of day t
    for lag in [1, 2, 5, 10, 20]:
        feat[f"ret_lag_{lag}d"] = ret.shift(lag)

    # Rolling realised volatility
    for window in [5, 20, 60]:
        feat[f"vol_{window}d"] = ret.rolling(window).std()

    # Momentum (price-based, already lagged via shift)
    feat["momentum_5d"]  = close.shift(1) / close.shift(6)  - 1
    feat["momentum_20d"] = close.shift(1) / close.shift(21) - 1
    feat["momentum_60d"] = close.shift(1) / close.shift(61) - 1

    # RSI on lagged close so no look-ahead
    feat["rsi_14"] = compute_rsi(close.shift(1))

    # Volume z-score (20-day)
    vol_mean = volume.shift(1).rolling(20).mean()
    vol_std  = volume.shift(1).rolling(20).std()
    feat["volume_zscore"] = (volume.shift(1) - vol_mean) / vol_std.replace(0, np.nan)

    # Earnings event flag — 1 in the 3-day window around any earnings announcement.
    # Captures the elevated volatility regime without requiring a surprise estimate.
    # NOTE: replace with actual earnings_surprise = (actual - consensus)/|consensus|
    # once you have point-in-time consensus data.
    feat["earnings_event"] = 0
    if earnings_dates is not None and len(earnings_dates) > 0:
        for ed in earnings_dates:
            window_start = ed - pd.Timedelta(days=1)
            window_end   = ed + pd.Timedelta(days=1)
            mask = (feat.index >= window_start) & (feat.index <= window_end)
            feat.loc[mask, "earnings_event"] = 1

    # Target: FORWARD_DAYS forward return — computed from future prices.
    # This column is the label; it must be dropped from X before training.
    feat["target"] = close.shift(-FORWARD_DAYS) / close - 1

    return feat.dropna()


# -- Walk-forward validation ---------------------------------------------------

def walk_forward(feat_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, XGBRegressor, list[str]]:
    """
    Expanding-window walk-forward validation.
    Trains on all data up to year Y, tests on year Y+1, then expands.
    Returns a summary DataFrame, a predictions DataFrame, the final fitted model, and feature names.
    """
    feature_cols = [c for c in feat_df.columns if c != "target"]
    years        = sorted(feat_df.index.year.unique())
    records      = []
    all_preds    = []
    model        = None

    for i in range(MIN_TRAIN_YRS - 1, len(years) - 1):
        train_cutoff = years[i]
        test_year    = years[i + 1]

        train = feat_df[feat_df.index.year <= train_cutoff]
        test  = feat_df[feat_df.index.year == test_year]

        if len(train) < 120 or len(test) < 20:
            continue

        model = XGBRegressor(
            n_estimators     = 300,
            max_depth        = 4,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            random_state     = 42,
            verbosity        = 0,
        )
        model.fit(train[feature_cols], train["target"])

        preds   = model.predict(test[feature_cols])
        actuals = test["target"].values

        records.append({
            "train_through"        : train_cutoff,
            "test_year"            : test_year,
            "n_train"              : len(train),
            "n_test"               : len(test),
            "mae"                  : mean_absolute_error(actuals, preds),
            "directional_accuracy" : float(np.mean(np.sign(preds) == np.sign(actuals))),
        })

        all_preds.append(pd.DataFrame({
            "date"            : test.index,
            "predicted_return": preds,
            "actual_return"   : actuals,
            "direction_correct": np.sign(preds) == np.sign(actuals),
        }))

    pred_df = pd.concat(all_preds).set_index("date") if all_preds else pd.DataFrame()
    return pd.DataFrame(records), pred_df, model, feature_cols


# -- Main ----------------------------------------------------------------------

def main():
    print(f"Downloading OHLCV data for {TICKERS} ({START_DATE} to {END_DATE})...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    all_results     = []
    all_importances = {}

    for ticker in TICKERS:
        print(f"\n{'-'*50}")
        print(f"  {ticker}")
        print(f"{'-'*50}")

        # Extract single-ticker slice from MultiIndex columns
        prices = raw.xs(ticker, level=1, axis=1)[["Close", "Volume"]].dropna()

        # Fetch earnings dates for the event flag
        try:
            ed_raw = yf.Ticker(ticker).earnings_dates
            earnings_dates = pd.DatetimeIndex(
                ed_raw.index.tz_localize(None) if ed_raw.index.tz is not None else ed_raw.index
            )
        except Exception:
            earnings_dates = pd.DatetimeIndex([])

        feat_df = build_features(prices, earnings_dates)
        print(f"  Feature matrix: {feat_df.shape[0]} rows x {feat_df.shape[1]} columns")

        results, pred_df, model, feature_cols = walk_forward(feat_df)

        if results.empty:
            print("  Not enough data for walk-forward validation.")
            continue

        results["ticker"] = ticker
        pred_df["ticker"] = ticker
        all_results.append(results)

        all_importances[ticker] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print("\n  Walk-forward results:")
        print(results[["test_year", "n_train", "n_test", "mae", "directional_accuracy"]]
              .to_string(index=False))

        print(f"\n  Sample predictions (5-day forward return, most recent 10 days):")
        sample = pred_df.tail(10)[["predicted_return", "actual_return", "direction_correct"]].copy()
        sample["predicted_return"] = sample["predicted_return"].map("{:+.2%}".format)
        sample["actual_return"]    = sample["actual_return"].map("{:+.2%}".format)
        print(sample.to_string())

    if not all_results:
        print("\nNo results — check data availability.")
        return

    # -- Aggregate summary -----------------------------------------------------
    combined = pd.concat(all_results, ignore_index=True)
    print(f"\n{'='*50}")
    print("  AGGREGATE RESULTS (all tickers, all years)")
    print(f"{'='*50}")
    print(f"  Mean MAE                : {combined['mae'].mean():.4f}")
    print(f"  Mean Directional Acc.   : {combined['directional_accuracy'].mean():.3f}  "
          f"(0.50 = random)")

    by_ticker = combined.groupby("ticker")[["mae", "directional_accuracy"]].mean()
    print("\n  Per-ticker averages:")
    print(by_ticker.round(4).to_string())

    # -- Feature importance ----------------------------------------------------
    avg_imp = (pd.concat(all_importances.values(), axis=1)
               .mean(axis=1)
               .sort_values(ascending=False))
    print(f"\n{'-'*50}")
    print("  AVERAGE FEATURE IMPORTANCE (across tickers)")
    print(f"{'-'*50}")
    print(avg_imp.round(4).to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
