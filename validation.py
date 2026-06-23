import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Callable


def walk_forward(
    feat_df: pd.DataFrame,
    min_train_yrs: int,
    model_factory: Callable,
) -> tuple[pd.DataFrame, pd.DataFrame, object, list[str]]:
    """
    Expanding-window walk-forward validation.
    Trains on all data up to year Y, tests on year Y+1, then expands.

    Works for both single-ticker and stacked cross-sectional DataFrames.
    If a 'ticker' column is present it is excluded from features but carried
    through into the predictions DataFrame for per-ticker evaluation.

    Returns:
        summary_df   : per-window MAE and directional accuracy
        pred_df      : date-indexed predicted vs actual returns (all windows combined)
        model        : the final fitted model (trained on the last window)
        feature_cols : list of feature column names used for training
    """
    EXCLUDE = {"target", "ticker"}
    feature_cols = [c for c in feat_df.columns if c not in EXCLUDE]
    years        = sorted(feat_df.index.year.unique())
    records      = []
    all_preds    = []
    model        = None

    for i in range(min_train_yrs - 1, len(years) - 1):
        train_cutoff = years[i]
        test_year    = years[i + 1]

        train = feat_df[feat_df.index.year <= train_cutoff].dropna(subset=feature_cols)
        test  = feat_df[feat_df.index.year == test_year].dropna(subset=feature_cols)

        if len(train) < 120 or len(test) < 20:
            continue

        model = model_factory()
        model.fit(train[feature_cols], train["target"])

        preds   = model.predict(test[feature_cols])
        actuals = test["target"].values

        records.append({
            "train_through"        : train_cutoff,
            "test_year"            : test_year,
            "n_train"              : len(train),
            "n_test"               : len(test),
            "mae"                  : mean_absolute_error(actuals, preds),
            "directional_accuracy" : float(np.mean(np.sign(preds) == np.sign(actuals))),
        })

        pred_row = {
            "date"            : test.index,
            "predicted_return": preds,
            "actual_return"   : actuals,
        }
        if "ticker" in test.columns:
            pred_row["ticker"] = test["ticker"].values

        all_preds.append(pd.DataFrame(pred_row))

    pred_df = pd.concat(all_preds).set_index("date") if all_preds else pd.DataFrame()
    return pd.DataFrame(records), pred_df, model, feature_cols
