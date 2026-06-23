import warnings
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

MODEL_NAME = "ARIMA"

# (p, d, q) — AR order, differencing, MA order
# Returns are already stationary so d=0. AR(2) + MA(1) is a sensible default.
DEFAULT_ORDER = (2, 0, 1)


def walk_forward_arima(
    stacked: pd.DataFrame,
    min_train_yrs: int,
    order: tuple = DEFAULT_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-ticker expanding-window ARIMA walk-forward.

    For each year-fold and each ticker, fits ARIMA on that ticker's training
    return series and forecasts the test period. Returns (summary_df, pred_df)
    in the same format as validation.walk_forward so compute_metrics works
    unchanged.
    """
    years   = sorted(stacked.index.year.unique())
    tickers = stacked["ticker"].unique()
    records = []
    all_preds = []

    for i in range(min_train_yrs - 1, len(years) - 1):
        train_cutoff = years[i]
        test_year    = years[i + 1]

        fold_preds = []

        for ticker in tickers:
            tkr = stacked[stacked["ticker"] == ticker].sort_index()

            train_series = tkr.loc[tkr.index.year <= train_cutoff, "target"].dropna()
            test_rows    = tkr.loc[tkr.index.year == test_year].dropna(subset=["target"])

            # Need enough history for ARIMA to fit
            if len(train_series) < 60 or len(test_rows) == 0:
                continue

            try:
                # reset_index strips the DatetimeIndex so statsmodels uses integer
                # indexing — eliminates the "no frequency" warnings entirely.
                # catch_warnings suppresses the convergence warning on noisy tickers.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = ARIMA(train_series.reset_index(drop=True), order=order).fit(maxiter=200)
                preds  = fit.forecast(steps=len(test_rows)).values
                actual = test_rows["target"].values

                fold_preds.append(pd.DataFrame({
                    "date"            : test_rows.index,
                    "predicted_return": preds,
                    "actual_return"   : actual,
                    "ticker"          : ticker,
                }))
            except Exception:
                # ARIMA can fail to converge on thin/noisy series — skip that ticker
                continue

        if not fold_preds:
            continue

        fold_df = pd.concat(fold_preds, ignore_index=True)
        records.append({
            "train_through"       : train_cutoff,
            "test_year"           : test_year,
            "n_train"             : int((stacked.index.year <= train_cutoff).sum()),
            "n_test"              : int((stacked.index.year == test_year).sum()),
            "mae"                 : mean_absolute_error(fold_df["actual_return"], fold_df["predicted_return"]),
            "directional_accuracy": float(np.mean(np.sign(fold_df["predicted_return"]) == np.sign(fold_df["actual_return"]))),
        })
        all_preds.append(fold_df)

    pred_df = pd.concat(all_preds).set_index("date") if all_preds else pd.DataFrame()
    return pd.DataFrame(records), pred_df
