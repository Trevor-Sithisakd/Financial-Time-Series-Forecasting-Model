"""
Optuna hyperparameter search.

Each model module defines its own param_space(trial) and build_from_params(params).
Pass whichever active_model you are already using in main.py.
"""

import json
import optuna
import pandas as pd
from pathlib import Path

from config import MIN_TRAIN_YRS, FORWARD_DAYS
from validation import walk_forward
from evaluation import compute_metrics


optuna.logging.set_verbosity(optuna.logging.WARNING)

_CACHE_DIR = Path("tuned_params")


def run_study(
    active_model,
    stacked: pd.DataFrame,
    n_trials: int = 50,
    tune_years: int = 4,
    retune: bool = False,
) -> dict:
    cache_file = _CACHE_DIR / f"{active_model.MODEL_NAME}_params.json"

    if not retune and cache_file.exists():
        params = json.loads(cache_file.read_text())
        print(f"Loaded cached params for {active_model.MODEL_NAME} (pass retune=True to re-run)")
        return params

    # Slice to recent history only — keeps each trial fast without random-sampling the time series
    dates = stacked.index.get_level_values("Date")
    cutoff = dates.max() - pd.DateOffset(years=tune_years)
    tune_data = stacked[dates >= cutoff]

    print(f"\nTuning {active_model.MODEL_NAME} on last {tune_years}y — {n_trials} trials")

    def objective(trial: optuna.Trial) -> float:
        params = active_model.param_space(trial)
        _, pred_df, _, _ = walk_forward(
            tune_data, MIN_TRAIN_YRS,
            lambda: active_model.build_from_params(params),
        )
        if pred_df.empty:
            return -1.0
        return compute_metrics(pred_df, FORWARD_DAYS)["ic"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best IC     : {study.best_value:+.4f}")
    print(f"Best params : {study.best_params}")

    _CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(study.best_params, indent=2))

    return study.best_params
