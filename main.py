import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import yfinance as yf

from config import TICKERS, START_DATE, END_DATE, FORWARD_DAYS, MIN_TRAIN_YRS, MODEL_PARAMS
from features import build_features
from validation import walk_forward
from reporting import print_ticker_results, print_aggregate_summary, print_feature_importance


def fetch_earnings_dates(ticker: str) -> pd.DatetimeIndex:
    try:
        ed_raw = yf.Ticker(ticker).earnings_dates
        idx = ed_raw.index
        return pd.DatetimeIndex(idx.tz_localize(None) if idx.tz is not None else idx)
    except Exception:
        return pd.DatetimeIndex([])


def main():
    print(f"Downloading OHLCV data for {TICKERS} ({START_DATE} to {END_DATE})...")
    raw = yf.download(TICKERS, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    all_results     = []
    all_importances = {}

    for ticker in TICKERS:
        prices         = raw.xs(ticker, level=1, axis=1)[["Close", "Volume"]].dropna()
        earnings_dates = fetch_earnings_dates(ticker)
        feat_df        = build_features(prices, earnings_dates, FORWARD_DAYS)

        results, pred_df, model, feature_cols = walk_forward(feat_df, MIN_TRAIN_YRS, MODEL_PARAMS)

        if results.empty:
            print(f"\n  {ticker}: not enough data for walk-forward validation.")
            continue

        results["ticker"] = ticker
        pred_df["ticker"] = ticker
        all_results.append(results)
        all_importances[ticker] = pd.Series(
            model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

        print_ticker_results(ticker, feat_df.shape, results, pred_df)

    if not all_results:
        print("\nNo results — check data availability.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    print_aggregate_summary(combined)
    print_feature_importance(all_importances)
    print("\nDone.")


if __name__ == "__main__":
    main()
