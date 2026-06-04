import numpy as np
import pandas as pd
from scipy import stats


def information_coefficient(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Spearman rank correlation between predicted and realised returns."""
    corr, _ = stats.spearmanr(predicted, actual)
    return float(corr)


def rank_ic(predicted: np.ndarray, actual: np.ndarray) -> float:
    """IC computed explicitly on integer ranks."""
    pred_ranks  = pd.Series(predicted).rank().values
    actual_ranks = pd.Series(actual).rank().values
    corr, _ = stats.spearmanr(pred_ranks, actual_ranks)
    return float(corr)


def directional_accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
    return float(np.mean(np.sign(predicted) == np.sign(actual)))


def top_decile_mean_return(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Mean actual return of the top-10% predicted days."""
    n   = max(1, len(predicted) // 10)
    idx = np.argsort(predicted)[-n:]
    return float(np.mean(actual[idx]))


def top_decile_sharpe(predicted: np.ndarray, actual: np.ndarray, forward_days: int = 5) -> float:
    """
    Annualised Sharpe of a strategy that goes long on top-decile prediction days.
    Uses forward_days to set the number of non-overlapping periods per year.
    """
    n       = max(1, len(predicted) // 10)
    idx     = np.argsort(predicted)[-n:]
    returns = actual[idx]
    std     = returns.std()
    if std == 0:
        return 0.0
    periods_per_year = 252 / forward_days
    return float((returns.mean() / std) * np.sqrt(periods_per_year))


def compute_metrics(pred_df: pd.DataFrame, forward_days: int) -> dict:
    """Compute all quant metrics from a predictions DataFrame."""
    predicted = pred_df["predicted_return"].values
    actual    = pred_df["actual_return"].values
    return {
        "ic"                  : information_coefficient(predicted, actual),
        "rank_ic"             : rank_ic(predicted, actual),
        "directional_accuracy": directional_accuracy(predicted, actual),
        "top_decile_return"   : top_decile_mean_return(predicted, actual),
        "sharpe"              : top_decile_sharpe(predicted, actual, forward_days),
    }
