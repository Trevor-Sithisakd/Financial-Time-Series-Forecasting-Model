# testing if build features only uses price data from previous dates and does not future data in predictions
import numpy as np
import pandas as pd 
import pytest 
from features import build_features
from validation import walk_forward
from sklearn.linear_model import Ridge

# create price data 
def create_prices(n=200, start="2020-01-01"):
    dates = pd.bdate_range(start=start, periods=n)
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5), index=dates)
    return pd.DataFrame({
        "Close": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)

# target leakage
def test_target_not_in_features():
    prices = create_prices(200)
    features = build_features(prices, pd.DatetimeIndex([]), forward_days=5)
    feature_cols = [c for c in features.columns if c != "target"]

    assert "target" not in feature_cols

def test_momentum():
    prices = create_prices(200)
    features = build_features(prices, pd.DatetimeIndex([]), forward_days=5)
    close = prices["Close"]

    for date in features.index[10:15]:
        expected = close.shift(1).loc[date] / close.shift(6).loc[date] - 1
        actual = features.loc[date, "momentum_5d"]
        assert abs(actual- expected) < 1e-9, f"Leakage detected on {date}"

def test_walk_foward_overlap():
    prices = create_prices(900, start="2018-01-01")
    spy = create_prices(900 , start="2018-01-01")["Close"].rename("SPY")
    features = build_features(prices, pd.DatetimeIndex([]), forward_days=5,
                             benchmark_close=spy)
    _, pred_df, _, _ = walk_forward(features, min_train_yrs=2, model_factory=Ridge)
    years = sorted(features.index.year.unique())
    print(f"years found: {years}")  
    print(f"number of years: {len(years)}")
    min_test_year = years[2] # makes sure that the valid test year is the third year

    assert pred_df.index.min().year >= min_test_year