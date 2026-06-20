from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "RandomForest"


def build():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )


def build_from_params(params: dict):
    return RandomForestRegressor(**params)


def param_space(trial) -> dict:
    max_depth_choice = trial.suggest_categorical("max_depth_choice", ["limited", "unlimited"])
    max_depth = trial.suggest_int("max_depth", 3, 20) if max_depth_choice == "limited" else None

    return {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 800),
        "max_depth"        : max_depth,
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features"     : trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
        "bootstrap"        : trial.suggest_categorical("bootstrap", [True, False]),
        "random_state"     : 42,
        "n_jobs"           : -1,
    }