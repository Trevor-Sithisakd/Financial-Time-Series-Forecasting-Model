from xgboost import XGBRegressor

MODEL_NAME = "XGBoost"

_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state"    : 42,
    "verbosity"       : 0,
}


def build():
    return XGBRegressor(**_PARAMS)
