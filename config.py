MLFLOW_EXPERIMENT = "financial-forecasting"

TICKERS       = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
START_DATE    = "2018-01-01"
END_DATE      = "2024-12-31"
FORWARD_DAYS  = 5   # trading days ahead to predict
MIN_TRAIN_YRS = 3   # minimum years of history before first test window

MODEL_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state"    : 42,
    "verbosity"       : 0,
}
