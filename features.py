import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """Returns (signal_line, histogram). Histogram = MACD line - signal line."""
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return signal_line, macd_line - signal_line


def compute_bollinger_position(series: pd.Series, window: int = 20) -> pd.Series:
    """Price position within Bollinger Bands: 0 = lower band, 0.5 = middle, 1 = upper band."""
    sma   = series.rolling(window).mean()
    std   = series.rolling(window).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    return (series - lower) / (upper - lower).replace(0, np.nan)


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
) -> pd.Series:
    """Average True Range — typical daily price range, used to normalise returns."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def add_cross_sectional_features(feat_dict: dict) -> dict:
    """
    Add cross-sectional rank and peer-relative features to each ticker's DataFrame.
    Must be called after all tickers have been processed by build_features so their
    feature columns are comparable on the same dates.

    Features added per ticker:
      xs_{col}_rank      : percentile rank (0=worst, 1=best) of that feature
                           among all tickers on each date
      xs_peer_excess_ret : 5-day return of this ticker minus the average of peers
    """
    tickers  = list(feat_dict.keys())
    rank_cols = ["momentum_20d", "momentum_60d", "rsi_14", "vol_20d"]

    for col in rank_cols:
        # Build date x ticker matrix; pandas aligns on index automatically
        combined = pd.DataFrame({
            t: feat_dict[t][col]
            for t in tickers
            if col in feat_dict[t].columns
        })
        ranked = combined.rank(axis=1, pct=True)
        for ticker in tickers:
            if ticker in ranked.columns:
                feat_dict[ticker][f"xs_{col}_rank"] = ranked[ticker]

    # Peer excess return: how did this ticker do vs the average of the other 4?
    ret_5d = pd.DataFrame({
        t: feat_dict[t]["momentum_5d"]
        for t in tickers
        if "momentum_5d" in feat_dict[t].columns
    })
    peer_avg = ret_5d.mean(axis=1)
    for ticker in tickers:
        if ticker in ret_5d.columns:
            feat_dict[ticker]["xs_peer_excess_ret"] = ret_5d[ticker] - peer_avg

    return feat_dict


def build_features(
    prices: pd.DataFrame,
    earnings_dates: pd.DatetimeIndex,
    forward_days: int,
    benchmark_close: pd.Series = None,
    earnings_surprise: pd.Series = None,
    macro_data: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Build a feature matrix from a single-ticker OHLCV DataFrame.
    All features use shift(1) so only yesterday's data is available at prediction time.
    The 'target' column is the label — drop from X before training.

    Args:
        benchmark_close   : SPY close — target becomes excess return over SPY.
        earnings_surprise : Date-indexed surprise values; applied over 5-day PEAD window.
        macro_data        : Date-indexed macro features (VIX, yields, SPY momentum).
                            Lagged 1 day inside this function.
    """
    close  = prices["Close"]
    volume = prices["Volume"]
    ret    = close.pct_change()
    lagged = close.shift(1)

    feat = pd.DataFrame(index=prices.index)

    # Lag returns
    for lag in [1, 2, 5, 10, 20]:
        feat[f"ret_lag_{lag}d"] = ret.shift(lag)

    # Rolling realised volatility
    for window in [5, 20, 60]:
        feat[f"vol_{window}d"] = ret.rolling(window).std()

    # Momentum — Jegadeesh & Titman cross-sectional factors
    feat["momentum_5d"]  = lagged / close.shift(6)  - 1
    feat["momentum_20d"] = lagged / close.shift(21) - 1
    feat["momentum_60d"] = lagged / close.shift(61) - 1

    # RSI
    feat["rsi_14"] = compute_rsi(lagged)

    # Volume z-score
    vol_mean = volume.shift(1).rolling(20).mean()
    vol_std  = volume.shift(1).rolling(20).std()
    feat["volume_zscore"] = (volume.shift(1) - vol_mean) / vol_std.replace(0, np.nan)

    # MACD
    macd_signal, macd_hist = compute_macd(lagged)
    feat["macd_signal"] = macd_signal
    feat["macd_hist"]   = macd_hist

    # Bollinger Band position
    feat["bb_position"] = compute_bollinger_position(lagged)

    # ATR-normalised return
    if "High" in prices.columns and "Low" in prices.columns:
        atr = compute_atr(prices["High"].shift(1), prices["Low"].shift(1), lagged)
        feat["atr_norm_ret"] = ret.shift(1) / atr.replace(0, np.nan)

    # Macro features — lagged 1 day so yesterday's macro is used at prediction time.
    # VIX tells the model what fear regime it's operating in.
    # Yield curve tells the model the macro headwind/tailwind for growth stocks.
    # SPY momentum tells the model whether the broad market is trending.
    if macro_data is not None:
        macro = macro_data.reindex(prices.index, method="ffill").shift(1)
        for col in macro.columns:
            feat[col] = macro[col]
    # Earnings features
    feat["earnings_surprise"] = 0.0
    feat["earnings_event"]    = 0

    if earnings_surprise is not None and not earnings_surprise.empty:
        for date, surprise_val in earnings_surprise.items():
            pead_start = date + pd.Timedelta(days=1)
            pead_end   = date + pd.Timedelta(days=8)
            mask = (feat.index >= pead_start) & (feat.index <= pead_end)
            feat.loc[mask, "earnings_surprise"] = float(surprise_val)
            feat.loc[mask, "earnings_event"]    = 1
    elif earnings_dates is not None and len(earnings_dates) > 0:
        for ed in earnings_dates:
            mask = (feat.index >= ed - pd.Timedelta(days=1)) & \
                   (feat.index <= ed + pd.Timedelta(days=1))
            feat.loc[mask, "earnings_event"] = 1

    # Target: excess return over benchmark, or raw forward return
    raw_fwd = close.shift(-forward_days) / close - 1
    if benchmark_close is not None:
        bench     = benchmark_close.sort_index().reindex(close.index, method="ffill")
        bench_fwd = bench.shift(-forward_days) / bench - 1
        feat["target"] = raw_fwd - bench_fwd
    else:
        feat["target"] = raw_fwd

    return feat.dropna()
