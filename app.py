"""
Streamlit dashboard for the 5-day excess-return forecasting pipeline.

Run:
    python export_dashboard_data.py   # once, generates dashboard_data/
    streamlit run app.py

The app only reads pre-generated CSVs — no training or inference happens here.
Page 1: model leaderboard (the hero) + stock universe table.
Page 2: per-ticker detail with model agreement and latest feature values.
"""

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = "dashboard_data"
IC_THRESHOLD = 0.05

st.set_page_config(
    page_title="Excess Return Forecasting — S&P 500",
    page_icon="📈",
    layout="wide",
)

# ── Data loading ──────────────────────────────────────────────────────────────

REQUIRED = ["model_metrics.csv", "fold_metrics.csv", "predictions.csv",
            "per_ticker_metrics.csv", "latest_features.csv", "market_caps.csv"]


@st.cache_data
def load_data():
    d = {}
    d["model_metrics"] = pd.read_csv(f"{DATA_DIR}/model_metrics.csv")
    d["fold_metrics"] = pd.read_csv(f"{DATA_DIR}/fold_metrics.csv")
    d["predictions"] = pd.read_csv(f"{DATA_DIR}/predictions.csv", parse_dates=["as_of_date"])
    d["per_ticker"] = pd.read_csv(f"{DATA_DIR}/per_ticker_metrics.csv")
    d["features"] = pd.read_csv(f"{DATA_DIR}/latest_features.csv", parse_dates=["date"])
    d["market_caps"] = pd.read_csv(f"{DATA_DIR}/market_caps.csv")
    return d


missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA_DIR, f))]
if missing:
    st.error(
        f"Missing data files in `{DATA_DIR}/`: {', '.join(missing)}.\n\n"
        "Run `python export_dashboard_data.py` first to generate them."
    )
    st.stop()

data = load_data()
MODELS = list(data["model_metrics"]["model_name"])
BEST_MODEL = data["model_metrics"].loc[data["model_metrics"]["ic"].idxmax(), "model_name"]
BEST_IC = data["model_metrics"]["ic"].max()
AS_OF = data["predictions"]["as_of_date"].max().date()

# ── Session state ─────────────────────────────────────────────────────────────

if "page" not in st.session_state:
    st.session_state.page = "universe"
if "ticker" not in st.session_state:
    st.session_state.ticker = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = BEST_MODEL


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt_cap(x: float) -> str:
    if pd.isna(x):
        return "—"
    if x >= 1e12:
        return f"${x / 1e12:.2f}T"
    return f"${x / 1e9:.0f}B"


def fmt_pct(x, signed=True) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:+.2%}" if signed else f"{x:.2%}"


def model_label(name: str) -> str:
    return f"{name}  ★ Best (IC: {BEST_IC:+.3f})" if name == BEST_MODEL else name


FEATURE_LABELS = {
    "rsi_14": "RSI (14d)",
    "macd_signal": "MACD signal line",
    "macd_hist": "MACD histogram",
    "bb_position": "Bollinger Band position (0–1)",
    "atr_norm_ret": "ATR-normalised return",
    "momentum_5d": "Momentum (1-week)",
    "momentum_20d": "Momentum (1-month)",
    "momentum_60d": "Momentum (3-month)",
    "vol_5d": "Realised vol (5d)",
    "vol_20d": "Realised vol (20d)",
    "vol_60d": "Realised vol (60d)",
    "volume_zscore": "Volume z-score (20d)",
    "vix_level": "VIX level",
    "vix_5d_chg": "VIX 5d change",
    "yield_curve": "Yield curve slope (10y − 3m)",
    "yield_curve_20d_chg": "Yield curve 20d change",
    "spy_ret_20d": "SPY 20d return",
    "spy_vol_20d": "SPY 20d realised vol",
    "xs_momentum_20d_rank": "Cross-sectional momentum rank (1m)",
    "xs_momentum_60d_rank": "Cross-sectional momentum rank (3m)",
    "xs_rsi_14_rank": "Cross-sectional RSI rank",
    "xs_vol_20d_rank": "Cross-sectional vol rank",
    "xs_peer_excess_ret": "Excess return vs peers (5d)",
}


def go_to_ticker(ticker: str):
    st.session_state.ticker = ticker
    st.session_state.page = "detail"


def go_home():
    st.session_state.page = "universe"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📈 Forecast Lab")
    st.caption(f"5-day **excess return vs SPY** · {len(data['market_caps'])} "
               f"large-cap tickers · adjusted close only")

    st.selectbox(
        "Model (drives universe table)",
        MODELS,
        format_func=model_label,
        key="selected_model",
    )

    st.divider()
    st.caption(
        f"Predictions are the latest **out-of-sample** walk-forward outputs "
        f"(as of {AS_OF}). Nothing is retrained in this app."
    )
    if st.session_state.page == "detail":
        st.button("← Back to universe", on_click=go_home, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Model evaluation + stock universe
# ══════════════════════════════════════════════════════════════════════════════

def page_universe():
    st.title("Model Evaluation — 5-Day Excess Return Forecasting")
    n_universe = len(data["market_caps"])
    st.caption(
        "Walk-forward (expanding window, 5-day embargo) evaluation of four models "
        f"predicting 5-day forward returns **in excess of SPY** for {n_universe} large-cap US equities. "
        "Engineered features: momentum, realised vol, RSI, MACD, Bollinger position, "
        "ATR-normalised returns, volume, macro regime (VIX, yield curve), cross-sectional ranks."
    )

    # ── Leaderboard ──
    st.subheader("Model leaderboard (pooled out-of-sample)")
    mm = data["model_metrics"].copy().sort_values("ic", ascending=False)
    lb = pd.DataFrame({
        "Model": [model_label(m) for m in mm["model_name"]],
        "IC": mm["ic"].map("{:+.4f}".format),
        "Sharpe (top-decile long)": mm["sharpe"].map("{:+.2f}".format),
        "Directional accuracy": mm["directional_accuracy"].map("{:.1%}".format),
        "Top-decile mean return": mm["top_decile_return"].map("{:+.2%}".format),
        "OOS obs": mm["n_obs"].map("{:,.0f}".format),
    })
    st.dataframe(lb, hide_index=True, use_container_width=True)

    best_row = mm.iloc[0]
    st.info(
        f"**Honest read:** the best model ({best_row['model_name']}, "
        f"IC {best_row['ic']:+.4f}) is still below the {IC_THRESHOLD} IC threshold for a "
        f"meaningful signal. The value of this project is the evaluation methodology and "
        f"the iteration story — not a tradeable strategy. Directional predictions below "
        f"should be read as model output, not investment signal.",
        icon="⚖️",
    )

    # ── IC by fold ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("IC by walk-forward test year")
        fm = data["fold_metrics"]
        fig = px.bar(
            fm, x="test_year", y="ic", color="model_name", barmode="group",
            labels={"test_year": "Test year", "ic": "Information coefficient",
                    "model_name": "Model"},
        )
        fig.add_hline(y=0, line_color="grey", line_width=1)
        fig.add_hline(y=IC_THRESHOLD, line_dash="dot", line_color="green",
                      annotation_text=f"signal threshold ({IC_THRESHOLD})")
        fig.update_layout(height=380, margin=dict(t=10, b=10),
                          legend=dict(orientation="h", y=-0.25))
        fig.update_xaxes(type="category")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Methodology")
        st.markdown(
            f"""
- **Target** — 5-day forward return minus SPY's (strips market beta; collapsed
  a spurious 54% directional accuracy down to ~50%)
- **Validation** — expanding-window walk-forward by year, with a **5-day embargo**
  between train and test so no forward-return window leaks
- **Leakage tests** — all features lagged ≥1 day; covered by `test_leakage.py`
- **Adjusted close only** — splits/dividends handled by yfinance `auto_adjust`
- **No retraining here** — every number is a pre-computed out-of-sample artefact
"""
        )

    st.divider()

    # ── Universe table ──
    st.subheader("Stock universe")
    st.caption(
        f"Latest out-of-sample predictions from **{st.session_state.selected_model}** "
        f"(as of {AS_OF}). Click a row for per-ticker model comparison."
    )

    preds = data["predictions"]
    model_preds = preds[preds["model_name"] == st.session_state.selected_model]
    caps = data["market_caps"].sort_values("market_cap", ascending=False).reset_index(drop=True)

    uni = caps.merge(model_preds, on="ticker", how="left")
    uni["Rank"] = np.arange(1, len(uni) + 1)
    uni["Market Cap"] = uni["market_cap"].map(fmt_cap)
    uni["Predicted Direction"] = uni["direction"].map({"up": "↑", "down": "↓"}).fillna("—")
    uni["Predicted Excess Return (5d)"] = uni["predicted_return"].map(
        lambda x: fmt_pct(x) if pd.notna(x) else "—")
    uni["Top Decile?"] = np.where(uni["decile_rank"] == 1, "✓", "—")
    uni.loc[uni["decile_rank"].isna(), "Top Decile?"] = "—"

    display = uni[["Rank", "company_name", "ticker", "Market Cap",
                   "Predicted Direction", "Predicted Excess Return (5d)", "Top Decile?"]]
    display = display.rename(columns={"company_name": "Company", "ticker": "Ticker"})

    event = st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
    )
    if event.selection.rows:
        go_to_ticker(uni.iloc[event.selection.rows[0]]["ticker"])
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Ticker detail
# ══════════════════════════════════════════════════════════════════════════════

def page_detail():
    ticker = st.session_state.ticker
    caps = data["market_caps"]
    row = caps[caps["ticker"] == ticker]
    name = row["company_name"].iloc[0] if not row.empty else ticker
    cap = fmt_cap(row["market_cap"].iloc[0]) if not row.empty else "—"

    st.button("← Back to universe", on_click=go_home)
    st.title(f"{name} ({ticker})")
    st.caption(f"Market cap: {cap} · predictions as of {AS_OF} · "
               f"target = 5-day return in excess of SPY")

    preds = data["predictions"]
    tp = preds[preds["ticker"] == ticker].set_index("model_name").reindex(MODELS)
    pt = data["per_ticker"]
    tm = pt[pt["ticker"] == ticker].set_index("model_name").reindex(MODELS)

    # ── Section 1: all model predictions ──
    st.subheader("Model predictions")
    if tp["predicted_return"].isna().all():
        st.warning(f"No predictions available for {ticker}.", icon="ℹ️")
    else:
        tbl = pd.DataFrame({
            "Model": [model_label(m) for m in MODELS],
            "Predicted Direction": tp["direction"].map({"up": "↑ up", "down": "↓ down"}).fillna("—"),
            "Predicted Excess Return (5d)": tp["predicted_return"].map(
                lambda x: fmt_pct(x) if pd.notna(x) else "—"),
            "Decile (1 = top)": tp["decile_rank"].map(
                lambda x: f"{int(x)}" if pd.notna(x) else "—"),
            "Top Decile?": np.where(tp["decile_rank"] == 1, "✓", "—"),
            "OOS IC (this ticker)": tm["ic"].map(
                lambda x: f"{x:+.3f}" if pd.notna(x) else "—"),
            "OOS dir. acc": tm["directional_accuracy"].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else "—"),
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)

    # ── Section 2: agreement chart ──
    st.subheader("Model agreement")
    valid = tp.dropna(subset=["predicted_return"])
    if valid.empty:
        st.caption("—")
    else:
        n_up = int((valid["predicted_return"] >= 0).sum())
        n_dn = len(valid) - n_up
        if n_up == len(valid):
            verdict = f"All {len(valid)} models are **bullish** on {ticker} vs SPY."
        elif n_dn == len(valid):
            verdict = f"All {len(valid)} models are **bearish** on {ticker} vs SPY."
        else:
            verdict = (f"Models **disagree**: {n_up} bullish vs {n_dn} bearish — "
                       f"a reminder of how weak the underlying signal is.")
        st.markdown(verdict)

        chart_df = valid.reset_index()
        chart_df["colour"] = np.where(chart_df["predicted_return"] >= 0,
                                      "bullish", "bearish")
        fig = px.bar(
            chart_df, x="predicted_return", y="model_name", orientation="h",
            color="colour",
            color_discrete_map={"bullish": "#2e7d32", "bearish": "#c62828"},
            labels={"predicted_return": "Predicted 5-day excess return",
                    "model_name": ""},
        )
        fig.add_vline(x=0, line_color="grey", line_width=1)
        fig.update_layout(height=280, showlegend=False,
                          margin=dict(t=10, b=10),
                          xaxis_tickformat="+.2%")
        st.plotly_chart(fig, use_container_width=True)

    # ── Section 3: latest feature values ──
    st.subheader("Latest feature values")
    feats = data["features"]
    frow = feats[feats["ticker"] == ticker]
    if frow.empty:
        st.caption("No feature data available for this ticker.")
    else:
        frow = frow.iloc[0]
        st.caption(f"As of {pd.to_datetime(frow['date']).date()} "
                   f"(all features lagged ≥1 day at build time)")
        items = [(label, frow[col]) for col, label in FEATURE_LABELS.items()
                 if col in frow.index and pd.notna(frow[col])]
        feat_tbl = pd.DataFrame(items, columns=["Feature", "Value"])
        feat_tbl["Value"] = feat_tbl["Value"].map(
            lambda x: f"{x:,.4f}" if isinstance(x, (int, float, np.floating)) else x)
        half = (len(feat_tbl) + 1) // 2
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(feat_tbl.iloc[:half], hide_index=True, use_container_width=True)
        with c2:
            st.dataframe(feat_tbl.iloc[half:], hide_index=True, use_container_width=True)


# ── Router ────────────────────────────────────────────────────────────────────

if st.session_state.page == "detail" and st.session_state.ticker:
    page_detail()
else:
    page_universe()
