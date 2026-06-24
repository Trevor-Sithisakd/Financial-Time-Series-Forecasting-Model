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


def build_from_params(params: dict):
    return XGBRegressor(**params)


def param_space(trial) -> dict:
    return {
        "n_estimators"    : trial.suggest_int("n_estimators", 100, 800),
        "learning_rate"   : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth"       : trial.suggest_int("max_depth", 3, 8),
        "subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma"           : trial.suggest_float("gamma", 1e-8, 0.1, log=True),
        "reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"    : 42,
        "verbosity"       : 0,
    }
