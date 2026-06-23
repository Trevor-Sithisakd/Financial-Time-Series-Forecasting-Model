import pandas as pd

# used for old per ticker testing
'''
def print_ticker_results(
    ticker: str,
    feat_shape: tuple,
    results: pd.DataFrame,
    pred_df: pd.DataFrame,
    n_sample: int = 10,
) -> None:
    print(f"\n{'-'*50}")
    print(f"  {ticker}")
    print(f"{'-'*50}")
    print(f"  Feature matrix: {feat_shape[0]} rows x {feat_shape[1]} columns")
    print("\n  Walk-forward results:")
    print(results[["test_year", "n_train", "n_test", "mae", "directional_accuracy"]]
          .to_string(index=False))

    print(f"\n  Sample predictions (most recent {n_sample} days):")
    sample = pred_df.tail(n_sample)[["predicted_return", "actual_return", "direction_correct"]].copy()
    sample["predicted_return"] = sample["predicted_return"].map("{:+.2%}".format)
    sample["actual_return"]    = sample["actual_return"].map("{:+.2%}".format)
    print(sample.to_string())

# old per ticker model testing code
def print_aggregate_summary(combined: pd.DataFrame) -> None:
    print(f"\n{'='*50}")
    print("  AGGREGATE RESULTS (all tickers, all years)")
    print(f"{'='*50}")
    print(f"  Mean MAE              : {combined['mae'].mean():.4f}")
    print(f"  Mean Directional Acc. : {combined['directional_accuracy'].mean():.3f}  "
          f"(0.50 = random)")
    print("\n  Per-ticker averages:")
    by_ticker = combined.groupby("ticker")[["mae", "directional_accuracy"]].mean()
    print(by_ticker.round(4).to_string())
'''

def print_feature_importance(all_importances: dict) -> None:
    avg_imp = (pd.concat(all_importances.values(), axis=1)
               .mean(axis=1)
               .sort_values(ascending=False))
    print(f"\n{'-'*50}")
    print("  AVERAGE FEATURE IMPORTANCE (across tickers)")
    print(f"{'-'*50}")
    print(avg_imp.round(4).to_string())


def print_quant_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{'='*50}")
    print(f"  QUANT METRICS — {model_name}")
    print(f"{'='*50}")
    print(f"  IC (Spearman)       : {metrics['ic']:+.4f}  (industry threshold ~0.05)")
    print(f"  Directional Acc.    : {metrics['directional_accuracy']:.3f}  (0.50 = random)")
    print(f"  Top-Decile Return   : {metrics['top_decile_return']:+.4f}  ({metrics['top_decile_return']:+.2%} per 5d)")
    print(f"  Top-Decile Sharpe   : {metrics['sharpe']:+.3f}  (>1.0 is good)")
