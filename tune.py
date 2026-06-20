"""
Optuna hyperparameter search.

Each model module defines its own param_space(trial) and build_from_params(params).
Pass whichever active_model you are already using in main.py.
"""

import optuna
import pandas as pd

from config import MIN_TRAIN_YRS, FORWARD_DAYS
from validation import walk_forward
from evaluation import compute_metrics


optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_study(active_model, stacked: pd.DataFrame, n_trials: int = 50) -> dict:
    print(f"\nTuning {active_model.MODEL_NAME} — {n_trials} trials")

    def objective(trial: optuna.Trial) -> float:
        params  = active_model.param_space(trial)
        _, pred_df, _, _ = walk_forward(
            stacked, MIN_TRAIN_YRS,
            lambda: active_model.build_from_params(params),
        )
        if pred_df.empty:
            return -1.0
        return compute_metrics(pred_df, FORWARD_DAYS)["ic"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best IC     : {study.best_value:+.4f}")
    print(f"Best params : {study.best_params}")
    return study.best_params
