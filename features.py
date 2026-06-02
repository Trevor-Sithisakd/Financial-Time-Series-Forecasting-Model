import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(
    prices: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex,
    forward_days: int,
) -> pd.DataFrame:
    """
    Build a feature matrix from a single-ticker OHLCV DataFrame.
    All features are lagged so they only use information available at time t.
    The 'target' column is the forward_days return label — drop from X before training.
    """
    close  = prices["Close"]
    volume = prices["Volume"]
    ret    = close.pct_change()

    feat = pd.DataFrame(index=prices.index)

    for lag in [1, 2, 5, 10, 20]:
        feat[f"ret_lag_{lag}d"] = ret.shift(lag)

    for window in [5, 20, 60]:
        feat[f"vol_{window}d"] = ret.rolling(window).std()

    feat["momentum_5d"]  = close.shift(1) / close.shift(6)  - 1
    feat["momentum_20d"] = close.shift(1) / close.shift(21) - 1
    feat["momentum_60d"] = close.shift(1) / close.shift(61) - 1

    feat["rsi_14"] = compute_rsi(close.shift(1))

    vol_mean = volume.shift(1).rolling(20).mean()
    vol_std  = volume.shift(1).rolling(20).std()
    feat["volume_zscore"] = (volume.shift(1) - vol_mean) / vol_std.replace(0, np.nan)

    # NOTE: replace with earnings_surprise = (actual - consensus) / |consensus|
    # once point-in-time consensus data is available.
    feat["earnings_event"] = 0
    if earnings_dates is not None and len(earnings_dates) > 0:
        for ed in earnings_dates:
            mask = (feat.index >= ed - pd.Timedelta(days=1)) & \
                   (feat.index <= ed + pd.Timedelta(days=1))
            feat.loc[mask, "earnings_event"] = 1

    feat["target"] = close.shift(-forward_days) / close - 1

    return feat.dropna()
