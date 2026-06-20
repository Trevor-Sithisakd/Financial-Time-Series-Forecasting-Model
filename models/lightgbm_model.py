from lightgbm import LGBMRegressor

MODEL_NAME = "LightGBM"


def build():
    return LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        verbose=-1,
    )


def build_from_params(params: dict):
    return LGBMRegressor(**params)


def param_space(trial) -> dict:
    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 800),
        "learning_rate"    : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth"        : trial.suggest_int("max_depth", 3, 8),
        "num_leaves"       : trial.suggest_int("num_leaves", 20, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state"     : 42,
        "verbose"          : -1,
    }