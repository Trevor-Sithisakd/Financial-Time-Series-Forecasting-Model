"""
Export flat data files for the Streamlit dashboard.

Runs the existing walk-forward pipeline for all four models and writes
everything the app needs into ./dashboard_data/. The app itself never
trains or runs inference — it only reads these CSVs.

Run once (or whenever the pipeline changes):
    python export_dashboard_data.py

Optional staged execution (used for resumable runs; results are identical):
    python export_dashboard_data.py --stage features
    python export_dashboard_data.py --stage model --model "XGBoost"
    python export_dashboard_data.py --stage finalize

Set DASH_OFFLINE=1 to skip all yfinance network calls (earnings features
default to 0, market caps fall back to a static snapshot).

Outputs (dashboard_data/):
    model_metrics.csv      one row per model: pooled OOS IC, Rank IC,
                           Sharpe (top-decile long), directional accuracy
    fold_metrics.csv       one row per model per walk-forward test year
    predictions.csv        latest out-of-sample prediction per (ticker, model):
                           predicted excess return, direction, decile rank
    per_ticker_metrics.csv per (ticker, model) OOS IC / Sharpe / dir-acc
    latest_features.csv    most recent feature row per ticker
    market_caps.csv        ticker, company_name, market_cap

Notes:
    - Predictions are the LAST out-of-sample walk-forward predictions
      (final test year), so nothing shown in the app was seen in training.
    - A 5-day embargo is applied: training rows whose 5-day forward
      target window overlaps the test year are dropped.
    - Target is excess return over SPY (adjusted close only).
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import re

import numpy as np
import pandas as pd

OFFLINE = os.environ.get("DASH_OFFLINE") == "1"

from config import TICKERS, FORWARD_DAYS, MIN_TRAIN_YRS
from features import build_features, add_cross_sectional_features
from evaluation import compute_metrics

# main.py imports mlflow at module level, but none of the loaders we use
# need it — stub it out if it isn't installed in this environment.
try:
    import mlflow  # noqa: F401
except ImportError:
    import sys
    import types
    sys.modules["mlflow"] = types.ModuleType("mlflow")

from main import (
    load_prices, load_spy, load_macro_data,
    fetch_earnings_dates, fetch_earnings_surprise, available_tickers,
)

import models.linear_baseline as m_lin
import models.xgboost_model as m_xgb
import models.random_forest as m_rf
import models.lightgbm_model as m_lgb

OUT_DIR = "dashboard_data"
EMBARGO_DAYS = FORWARD_DAYS  # drop train rows whose forward window leaks into test

MODELS = {
    "Ridge (linear baseline)": m_lin.build,
    "XGBoost": m_xgb.build,
    "Random Forest": m_rf.build,
    "LightGBM": m_lgb.build,
}

# Fallback market caps (approx USD billions) and names, used when yfinance
# is unreachable. Ordering is what matters for the dashboard.
STATIC_CAPS = {
    "AAPL": ("Apple", 3400), "NVDA": ("NVIDIA", 3300), "MSFT": ("Microsoft", 3300),
    "AMZN": ("Amazon", 2300), "GOOGL": ("Alphabet", 2100), "META": ("Meta Platforms", 1500),
    "TSLA": ("Tesla", 1000), "AVGO": ("Broadcom", 1100), "BRK-B": ("Berkshire Hathaway", 1000),
    "JPM": ("JPMorgan Chase", 700), "LLY": ("Eli Lilly", 800), "V": ("Visa", 600),
    "XOM": ("Exxon Mobil", 480), "MA": ("Mastercard", 520), "COST": ("Costco", 400),
    "UNH": ("UnitedHealth", 450), "NFLX": ("Netflix", 400), "HD": ("Home Depot", 380),
    "WMT": ("Walmart", 700), "PG": ("Procter & Gamble", 400), "JNJ": ("Johnson & Johnson", 380),
    "BAC": ("Bank of America", 320), "CRM": ("Salesforce", 280), "AMD": ("AMD", 260),
    "ORCL": ("Oracle", 450), "CVX": ("Chevron", 270), "MRK": ("Merck", 250),
    "ABBV": ("AbbVie", 320), "KO": ("Coca-Cola", 290), "WFC": ("Wells Fargo", 240),
    "CSCO": ("Cisco", 240), "ACN": ("Accenture", 200), "NOW": ("ServiceNow", 200),
    "IBM": ("IBM", 250), "MS": ("Morgan Stanley", 210), "GS": ("Goldman Sachs", 180),
    "PM": ("Philip Morris", 250), "PEP": ("PepsiCo", 230), "DIS": ("Disney", 200),
    "TMO": ("Thermo Fisher", 200), "INTU": ("Intuit", 190), "TXN": ("Texas Instruments", 170),
    "ISRG": ("Intuitive Surgical", 190), "AXP": ("American Express", 210), "AMGN": ("Amgen", 160),
    "GE": ("GE Aerospace", 240), "UBER": ("Uber", 180), "BKNG": ("Booking Holdings", 170),
    "QCOM": ("Qualcomm", 170), "RTX": ("RTX", 180), "CAT": ("Caterpillar", 170),
    "SPGI": ("S&P Global", 160), "BLK": ("BlackRock", 150), "LOW": ("Lowe's", 130),
    "HON": ("Honeywell", 140), "PFE": ("Pfizer", 140), "DE": ("Deere", 140),
    "UNP": ("Union Pacific", 140), "NEE": ("NextEra Energy", 150), "BSX": ("Boston Scientific", 150),
    "AMAT": ("Applied Materials", 140), "SYK": ("Stryker", 140), "MU": ("Micron", 130),
    "ADI": ("Analog Devices", 110), "PANW": ("Palo Alto Networks", 130), "ADP": ("ADP", 120),
    "REGN": ("Regeneron", 60), "GILD": ("Gilead", 140), "ETN": ("Eaton", 130),
    "LRCX": ("Lam Research", 120), "MMC": ("Marsh McLennan", 110), "BA": ("Boeing", 130),
    "VRTX": ("Vertex Pharma", 110), "CB": ("Chubb", 110), "SO": ("Southern Co", 100),
    "MCD": ("McDonald's", 210), "KLAC": ("KLA", 120), "CME": ("CME Group", 100),
    "CEG": ("Constellation Energy", 90), "BMY": ("Bristol Myers Squibb", 110),
    "CI": ("Cigna", 90), "PLD": ("Prologis", 100), "SNPS": ("Synopsys", 80),
    "CDNS": ("Cadence", 80), "MCO": ("Moody's", 90), "ICE": ("Intercontinental Exchange", 100),
    "WM": ("Waste Management", 90), "EOG": ("EOG Resources", 70), "CTAS": ("Cintas", 80),
    "ZTS": ("Zoetis", 70), "TJX": ("TJX", 140), "AON": ("Aon", 80),
    "APH": ("Amphenol", 110), "MSI": ("Motorola Solutions", 70), "MDLZ": ("Mondelez", 90),
    "PH": ("Parker Hannifin", 90), "NSC": ("Norfolk Southern", 60), "COF": ("Capital One", 80),
    "WELL": ("Welltower", 90), "ITW": ("Illinois Tool Works", 80),
}


def build_stacked_features() -> pd.DataFrame:
    """Same feature build as main.py, robust to yfinance being offline."""
    raw = load_prices()
    spy_close = load_spy()
    macro = load_macro_data(spy_close)

    feat_dfs = {}
    for ticker in available_tickers(raw):
        prices = raw.xs(ticker, level=1, axis=1)[["Close", "High", "Low", "Volume"]].dropna()
        if prices.empty:
            continue
        if OFFLINE:
            earn_dates, earn_surp = pd.DatetimeIndex([]), pd.Series(dtype=float)
        else:
            try:
                earn_dates = fetch_earnings_dates(ticker)
                earn_surp = fetch_earnings_surprise(ticker)
            except Exception:
                earn_dates, earn_surp = pd.DatetimeIndex([]), pd.Series(dtype=float)
        feat_dfs[ticker] = build_features(
            prices, earn_dates, FORWARD_DAYS,
            benchmark_close=spy_close,
            earnings_surprise=earn_surp,
            macro_data=macro,
        )
    feat_dfs = add_cross_sectional_features(feat_dfs)
    stacked = pd.concat(
        [df.assign(ticker=t) for t, df in feat_dfs.items()]
    ).sort_index()
    print(f"Stacked: {stacked.shape[0]:,} rows x {stacked.shape[1]} cols")
    return stacked


def walk_forward_embargo(feat_df: pd.DataFrame, model_factory) -> pd.DataFrame:
    """
    Expanding-window walk-forward (train <= year Y, test year Y+1) with a
    5-day embargo: the last EMBARGO_DAYS trading days of the training window
    are dropped so no training target overlaps the test period.
    Returns the pooled OOS predictions DataFrame (date-indexed,
    columns: ticker, predicted_return, actual_return).
    """
    feature_cols = [c for c in feat_df.columns if c not in {"target", "ticker"}]
    years = sorted(feat_df.index.year.unique())
    all_preds = []

    for i in range(MIN_TRAIN_YRS - 1, len(years) - 1):
        train_cutoff, test_year = years[i], years[i + 1]
        train = feat_df[feat_df.index.year <= train_cutoff].dropna(subset=feature_cols)
        test = feat_df[feat_df.index.year == test_year].dropna(subset=feature_cols)
        if len(train) < 120 or len(test) < 20:
            continue

        # 5-day embargo: drop train rows in the last EMBARGO_DAYS trading days
        train_dates = train.index.unique().sort_values()
        if len(train_dates) > EMBARGO_DAYS:
            cutoff_date = train_dates[-(EMBARGO_DAYS + 1)]
            train = train[train.index <= cutoff_date]

        model = model_factory()
        model.fit(train[feature_cols], train["target"])
        preds = model.predict(test[feature_cols])

        all_preds.append(pd.DataFrame({
            "date": test.index,
            "ticker": test["ticker"].values,
            "predicted_return": preds,
            "actual_return": test["target"].values,
        }))

    return pd.concat(all_preds).set_index("date")


def latest_predictions(pred_df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Latest OOS prediction per ticker, with cross-sectional decile rank."""
    last_date = pred_df.index.max()
    snap = pred_df[pred_df.index == last_date].copy()
    snap["model_name"] = model_name
    snap["direction"] = np.where(snap["predicted_return"] >= 0, "up", "down")
    # decile 1 = highest predicted return
    snap["decile_rank"] = (
        pd.qcut(snap["predicted_return"].rank(method="first", ascending=False),
                10, labels=range(1, 11)).astype(int)
    )
    snap = snap.reset_index().rename(columns={"date": "as_of_date"})
    return snap[["ticker", "model_name", "as_of_date",
                 "predicted_return", "actual_return", "direction", "decile_rank"]]


def _static_cap_row(t: str) -> dict:
    """Offline fallback row for one ticker. Curated caps for the leaders;
    everything else gets the symbol as its name and a small placeholder cap
    so the full universe still appears in the dashboard."""
    name, cap_b = STATIC_CAPS.get(t, (t, 10))  # 10 = ~$10B placeholder
    return {"ticker": t, "company_name": name,
            "market_cap": cap_b * 1e9, "source": "static_snapshot"}


def fetch_market_caps() -> pd.DataFrame:
    if OFFLINE:
        print("Market caps: static snapshot (DASH_OFFLINE=1)")
        return pd.DataFrame([_static_cap_row(t) for t in TICKERS])

    rows, failed = [], []
    try:
        import yfinance as yf
    except Exception as e:
        print(f"yfinance unavailable ({e}) — using static snapshot")
        return pd.DataFrame([_static_cap_row(t) for t in TICKERS])

    # Per-ticker resilience: a few invalid/delisted symbols shouldn't dump the
    # whole universe to the static snapshot — fall back only for the ones that
    # fail individually.
    for t in TICKERS:
        try:
            info = yf.Ticker(t).fast_info
            cap = info.get("market_cap") or info.get("marketCap")
            if not cap:
                raise ValueError("no cap")
            name = STATIC_CAPS.get(t, (t, None))[0]
            rows.append({"ticker": t, "company_name": name,
                         "market_cap": float(cap), "source": "yfinance"})
        except Exception:
            failed.append(t)
            rows.append(_static_cap_row(t))
    print(f"Market caps: live from yfinance "
          f"({len(TICKERS) - len(failed)}/{len(TICKERS)}; "
          f"{len(failed)} fell back to static)")
    return pd.DataFrame(rows)


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


STACKED_CACHE = f"{OUT_DIR}/_stacked.pkl"


def stage_features() -> pd.DataFrame:
    """Build (or load cached) stacked feature matrix."""
    if os.path.exists(STACKED_CACHE):
        print("Loading cached stacked features...")
        return pd.read_pickle(STACKED_CACHE)
    stacked = build_stacked_features()
    os.makedirs(OUT_DIR, exist_ok=True)
    stacked.to_pickle(STACKED_CACHE)
    return stacked


def stage_model(stacked: pd.DataFrame, name: str):
    """Run walk-forward for one model and write its intermediate CSVs."""
    slug = _slug(name)
    done_flag = f"{OUT_DIR}/_model_{slug}.done"
    if os.path.exists(done_flag):
        print(f"{name}: already done, skipping")
        return

    print(f"\n=== {name} ===")
    pred_df = walk_forward_embargo(stacked, MODELS[name])

    m = compute_metrics(pred_df, FORWARD_DAYS)
    m["model_name"] = name
    m["n_obs"] = len(pred_df)
    pd.DataFrame([m]).to_csv(f"{OUT_DIR}/_metrics_{slug}.csv", index=False)
    print({k: round(v, 4) for k, v in m.items() if isinstance(v, float)})

    fold_rows = []
    for year, grp in pred_df.groupby(pred_df.index.year):
        fm = compute_metrics(grp, FORWARD_DAYS)
        fm.update({"model_name": name, "test_year": int(year), "n_obs": len(grp)})
        fold_rows.append(fm)
    pd.DataFrame(fold_rows).to_csv(f"{OUT_DIR}/_folds_{slug}.csv", index=False)

    ticker_rows = []
    for ticker, grp in pred_df.groupby("ticker"):
        if len(grp) < 30:
            continue
        tm = compute_metrics(grp, FORWARD_DAYS)
        tm.update({"model_name": name, "ticker": ticker, "n_obs": len(grp)})
        ticker_rows.append(tm)
    pd.DataFrame(ticker_rows).to_csv(f"{OUT_DIR}/_ticker_{slug}.csv", index=False)

    latest_predictions(pred_df, name).to_csv(f"{OUT_DIR}/_preds_{slug}.csv", index=False)
    open(done_flag, "w").close()


def stage_finalize(stacked: pd.DataFrame):
    """Merge per-model intermediates into the final dashboard files."""
    slugs = [_slug(n) for n in MODELS]
    missing = [s for s in slugs if not os.path.exists(f"{OUT_DIR}/_model_{s}.done")]
    if missing:
        raise SystemExit(f"Models not yet run: {missing}")

    pd.concat([pd.read_csv(f"{OUT_DIR}/_metrics_{s}.csv") for s in slugs]) \
        .to_csv(f"{OUT_DIR}/model_metrics.csv", index=False)
    pd.concat([pd.read_csv(f"{OUT_DIR}/_folds_{s}.csv") for s in slugs]) \
        .to_csv(f"{OUT_DIR}/fold_metrics.csv", index=False)
    pd.concat([pd.read_csv(f"{OUT_DIR}/_ticker_{s}.csv") for s in slugs]) \
        .to_csv(f"{OUT_DIR}/per_ticker_metrics.csv", index=False)
    pd.concat([pd.read_csv(f"{OUT_DIR}/_preds_{s}.csv") for s in slugs]) \
        .to_csv(f"{OUT_DIR}/predictions.csv", index=False)

    feature_cols = [c for c in stacked.columns if c not in {"target", "ticker"}]
    latest_feats = stacked.sort_index().groupby("ticker").tail(1).reset_index()
    # the price index may be named 'Date' or unnamed depending on the cache
    latest_feats = latest_feats.rename(columns={latest_feats.columns[0]: "date"})
    latest_feats[["date", "ticker"] + feature_cols].to_csv(
        f"{OUT_DIR}/latest_features.csv", index=False)

    fetch_market_caps().to_csv(f"{OUT_DIR}/market_caps.csv", index=False)
    print(f"\nAll files written to {OUT_DIR}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["all", "features", "model", "finalize"],
                        default="all")
    parser.add_argument("--model", help="model name (for --stage model)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.stage == "features":
        stage_features()
        return
    if args.stage == "model":
        if args.model not in MODELS:
            raise SystemExit(f"Unknown model '{args.model}'. Options: {list(MODELS)}")
        stage_model(stage_features(), args.model)
        return
    if args.stage == "finalize":
        stage_finalize(stage_features())
        return

    # --stage all (default)
    stacked = stage_features()
    for name in MODELS:
        stage_model(stacked, name)
    stage_finalize(stacked)


if __name__ == "__main__":
    main()
# end of file
