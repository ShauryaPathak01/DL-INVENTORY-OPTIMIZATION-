"""
app.py - AI Inventory Optimization dashboard (Streamlit).

    streamlit run app.py

On first start it trains the whole pipeline on the built-in dummy data (~1 min on CPU) and caches the result in
``artifacts/results.pkl``. Use ``python train.py`` beforehand to train on M5 / other data, or the sidebar to retrain.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import uplift_at
from config import Config
from inventory import InventoryParams, compute_schedule, simulate_policy
from pipeline import MODELS, load_results, run_pipeline, save_results

st.set_page_config(page_title="AI Inventory Optimization", layout="wide", page_icon="📦")
ART = Path("artifacts") / "results.pkl"
COLORS = {"Naive": "#9aa0a6", "XGBoost": "#f29900", "LSTM": "#a142f4", "TFT-Sparse": "#1a73e8"}


# --------------------------------------------------------------------------- results loading
def _train(cfg: Config):
    bar = st.progress(0.0, text="Training models ...")
    msgs = []

    def log(m):
        msgs.append(m)
        bar.progress(min(0.95, len(msgs) / (2 * cfg.epochs + 4)), text=m)
    res = run_pipeline(cfg, log=log)
    save_results(res, ART)
    bar.empty()
    return res


if "res" not in st.session_state:
    if ART.exists():
        st.session_state.res = load_results(ART)
    else:
        with st.spinner("First run: training on dummy data (about a minute) ..."):
            st.session_state.res = _train(Config())
res = st.session_state.res
cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in res["cfg"].items()})
H = res["split"]["H"]
meta = res["meta"]

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📦 Inventory AI")
    st.caption(f"Source: **{cfg.source}** · {len(meta)} SKU-store series · horizon {H}d · device {res['device']}")
    st.subheader("Inventory policy")
    sl = st.slider("Target service level", 0.80, 0.995, float(cfg.service_level), 0.005)
    lt = st.slider("Lead time (days)", 1, 14, int(cfg.lead_time))
    rp = st.slider("Review period (days)", 1, 14, int(cfg.review_period))
    hold = st.number_input("Holding cost (% of unit cost / day)", 0.0, 5.0, cfg.holding_rate_per_day * 100, 0.05) / 100
    goodwill = st.slider("Stock-out goodwill multiplier", 0.0, 3.0, float(cfg.goodwill_mult), 0.1)
    pack = st.number_input("Pack size", 1, 100, 1)
    params = InventoryParams.from_config(cfg).with_(service_level=sl, lead_time=lt, review_period=rp,
                                                    holding_rate_per_day=hold, goodwill_mult=goodwill, pack_size=int(pack))
    st.subheader("Filters")
    stores = ["All"] + sorted(meta.store_id.unique())
    store_f = st.selectbox("Store", stores)
    with st.expander("Retrain"):
        src = st.selectbox("Data source", ["dummy", "m5", "walmart", "favorita"], index=["dummy", "m5", "walmart", "favorita"].index(cfg.source))
        ddir = st.text_input("Data dir (real datasets)", cfg.data_dir)
        ep = st.slider("Epochs", 2, 30, int(cfg.epochs))
        lam = st.slider("λ mechanism loss", 0.0, 0.5, float(cfg.lambda_mech), 0.01)
        if st.button("Train / retrain"):
            kw = cfg.to_dict() | {"source": src, "data_dir": ddir, "epochs": ep, "lambda_mech": lam}
            kw["quantiles"] = tuple(kw["quantiles"])
            st.session_state.res = _train(Config(**kw))
            st.rerun()

mask = np.ones(len(meta), bool) if store_f == "All" else (meta.store_id == store_f).values
fq = res["future"]["q"]["TFT-Sparse"]
sched = compute_schedule(meta, *fq, res["inv"]["on_hand"], res["inv"]["on_order"], res["inv"]["price"], params)
sched_f = sched[mask]

tabs = st.tabs(["📊 KPIs", "📈 Forecast", "🧮 Inventory plan", "🧪 What-if", "🔍 Explainability",
                "🚨 Anomalies & promotions", "⚙️ Model"])

# ============================================================================ KPIs
with tabs[0]:
    m = res["metrics"].set_index("model")
    t = m.loc["TFT-Sparse"]
    c = st.columns(5)
    c[0].metric("Forecast accuracy (1-WAPE)", f"{t['Forecast accuracy (1-WAPE)']:.1%}",
                f"{(t['Forecast accuracy (1-WAPE)'] - m.loc['XGBoost', 'Forecast accuracy (1-WAPE)']) * 100:+.1f} pp vs XGBoost")
    c[1].metric("Inventory cost (back-test)", f"{t['Inventory cost']:,.0f}")
    c[2].metric("Service level (in-stock)", f"{t['Service level (in-stock)']:.1%}")
    c[3].metric("Fill rate", f"{t['Fill rate']:.1%}")
    exp_rev = sched_f['expected_revenue'].sum() if ('expected_revenue' in sched_f and sched_f['expected_revenue'].sum() > 0) else float(np.sum(fq[1][mask] * res["inv"]["price"][mask, None]))
    c[4].metric("Expected revenue (next %dd)" % H, f"{exp_rev:,.0f}")

    st.caption("Accuracy & policy KPIs are measured on a held-out 28-day test period (two non-overlapping 14-day "
               "forecast blocks, the policy back-test covers the last block). Expected revenue is the forward-looking "
               "estimate from the schedule below.")
    st.info(f"📁 Active Dataset: **{cfg.source.upper()}** | Evaluated on {len(meta)} Series | Horizon: {H} Days")
    st.subheader("Model benchmark (test period)")
    show = res["metrics"].copy()
    min_cols = [c for c in ["MAE", "RMSE", "MASE", "WRMSSE", "Pinball loss (P10/50/90)", "Inventory cost"] if c in show.columns]
    max_cols = [c for c in ["Forecast accuracy (1-WAPE)", "Service level (in-stock)", "Fill rate", "Revenue"] if c in show.columns]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "model"})
                 .highlight_min(subset=min_cols, color="#d2f8d2")
                 .highlight_max(subset=max_cols, color="#d2f8d2"),
                 width="stretch", hide_index=True)


    a, b = st.columns(2)
    fig = px.bar(show, x="model", y="WRMSSE", color="model", color_discrete_map=COLORS, title="WRMSSE (lower is better)")
    a.plotly_chart(fig.update_layout(showlegend=False), width="stretch")
    fig = px.bar(show, x="model", y="Inventory cost", color="model", color_discrete_map=COLORS,
                 title="Inventory cost under the (s,S) policy (lower is better)")
    b.plotly_chart(fig.update_layout(showlegend=False), width="stretch")
    cov = res["metrics"].set_index("model")["P10-P90 coverage (nominal 0.80)"]
    st.info(f"P10–P90 coverage of the Transformer: **{cov['TFT-Sparse']:.1%}** (nominal 80%). "
            "Values well below 80% mean intervals are too narrow and safety stocks will under-protect.")

# ============================================================================ Forecast
with tabs[1]:
    hier = res["hierarchy"]
    labels = hier["labels"]
    kind = st.radio("Level", hier["level_names"], horizontal=True, index=4)
    li = hier["level_names"].index(kind)
    idx = [i for i, l in enumerate(hier["level"]) if l == li]
    opts = [labels[i] for i in idx]
    if kind == "Store x Item" and store_f != "All":
        opts = [o for o in opts if o.startswith(store_f)]
    node = st.selectbox("Node", opts)
    j = labels.index(node)
    hist = hier["agg_hist"][j]
    dates = pd.to_datetime(res["dates"])
    fd = pd.to_datetime(res["future_dates"])
    n_show = st.slider("History to show (days)", 30, 300, 90)
    f10, f50, f90 = (hier["agg_future"][k][j] for k in ("q10", "q50", "q90"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates[-n_show:], y=hist[-n_show:], name="Actual", line=dict(color="#202124")))
    fig.add_trace(go.Scatter(x=fd, y=f90, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fd, y=f10, fill="tonexty", fillcolor="rgba(26,115,232,0.2)", line=dict(width=0),
                             name="P10–P90"))
    fig.add_trace(go.Scatter(x=fd, y=f50, name="P50 forecast", line=dict(color="#1a73e8", width=3)))
    fig.update_layout(title=f"Demand forecast – {node}", height=430, hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.caption("Levels above SKU use bottom-up reconciliation: P50s add up exactly; the interval is aggregated assuming "
               "independent errors (a mild under-estimate when errors are positively correlated).")
    st.subheader("Back-test: forecast vs actual on the held-out test period")
    test = res["test"]
    if kind == "Store x Item":
        s_i = [i for i, sid in enumerate(res["series_ids"]) if sid.replace("__", " / ") == node][0]
        td = pd.to_datetime(test["dates"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=td, y=test["y"][s_i], name="Actual", line=dict(color="#202124")))
        for mname in MODELS:
            q = test["q"][mname]
            if mname == "TFT-Sparse":
                fig.add_trace(go.Scatter(x=td, y=q[2][s_i], line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=td, y=q[0][s_i], fill="tonexty", fillcolor="rgba(26,115,232,0.15)",
                                         line=dict(width=0), name="TFT P10–P90"))
            fig.add_trace(go.Scatter(x=td, y=q[1][s_i], name=mname, line=dict(color=COLORS[mname], dash="solid" if mname == "TFT-Sparse" else "dot")))
        fig.update_layout(height=400, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("Select the *Store x Item* level to compare the models on a single SKU.")

# ============================================================================ Inventory plan
with tabs[2]:
    c = st.columns(4)
    c[0].metric("SKUs to reorder now", int(sched_f["reorder_now"].sum()), f"of {len(sched_f)}")
    c[1].metric("Total suggested order (units)", f"{sched_f['suggested_order_qty_SOQ'].sum():,.0f}")
    c[2].metric("Avg stock-out risk", f"{sched_f['stockout_risk'].mean():.1%}")
    c[3].metric("Expected cost", f"{sched_f['expected_cost'].sum():,.1f}")
    only = st.checkbox("Show only SKUs that need an order", value=False)
    view = sched_f[sched_f.reorder_now] if only else sched_f
    st.dataframe(view.sort_values("stockout_risk", ascending=False), width="stretch", hide_index=True)
    st.download_button("⬇️ Download optimization schedule (CSV)", sched_f.to_csv(index=False).encode(),
                       "optimization_schedule.csv", "text/csv")
    fig = px.scatter(sched_f, x="stockout_risk", y="overstock_risk", size="expected_cost", color="store_id",
                     hover_data=["item_id", "suggested_order_qty_SOQ"], title="Risk map (bubble = expected cost)")
    st.plotly_chart(fig, width="stretch")
    with st.expander("How these numbers are computed"):
        st.markdown(f"""
* σ per day = (P90 − P10) / 2.563; window σ accounts for error autocorrelation ρ = {params.rho}.
* **SS** = z({params.service_level:.3f}) · σ_L · **ROP** = μ_L + SS (lead time L = {lt}d)
* **Order-up-to** = μ_(L+P) + z · σ_(L+P) (review period P = {rp}d) · **SOQ** = max(0, OUT − IP) if IP ≤ ROP
* Stock-out risk = P(D_L > IP); overstock risk = P(leftover > {params.overstock_threshold:.0%} of window demand)
* Expected cost = holding · E[leftover] · (L+P) + stock-out penalty · E[shortage] (normal loss function)
* *On-hand stock is simulated for the dummy demo – connect your ERP stock levels in `pipeline.py` (`on_hand`, `on_order`).*""")

# ============================================================================ What-if
with tabs[3]:
    st.write("Simulate the (s,S) policy day by day on the held-out last forecast block, using real demand.")
    bt = res["backtest"]
    c1, c2, c3 = st.columns(3)
    mdl = c1.selectbox("Forecast driving the policy", MODELS, index=3)
    uplift_mode = c2.selectbox("Promotion scenario", ["None", "Manual uplift", "From fitted discount curve"])
    aware = c3.checkbox("Forecast is aware of the promotion", value=True)
    uplift, win = 0.0, (0, 0)
    if uplift_mode != "None":
        win = st.slider("Promotion window (day index in the block)", 0, H, (3, 8))
        if uplift_mode == "Manual uplift":
            uplift = st.slider("Demand uplift", 0.0, 3.0, 0.5, 0.05)
        else:
            pr = res["promo"]
            if pr.get("ok"):
                depth = st.slider("Discount depth", 0.05, 0.6, 0.25, 0.05)
                uplift = uplift_at(depth, pr)
                st.caption(f"Fitted saturating uplift at {depth:.0%} discount: **+{uplift:.0%}** demand")
            else:
                st.warning("Not enough promotion history to fit a curve.")
    base_p = InventoryParams.from_config(cfg)
    mk = mask
    q = tuple(a[mk] for a in bt["q"][mdl])
    kw = dict(price=res["inv"]["price"][mk], init_on_hand=bt["init_on_hand"][mk])
    r_cur = simulate_policy(bt["demand"][mk], *q, params=params, promo_uplift=uplift, promo_window=win, forecast_aware=aware, **kw)
    r_base = simulate_policy(bt["demand"][mk], *q, params=base_p, **kw)
    cols = st.columns(4)
    for col, (lab, key, fmt, inv) in zip(cols, [("Inventory cost", "inventory_cost", "{:,.0f}", True),
                                                ("Service level", "service_level", "{:.1%}", False),
                                                ("Fill rate", "fill_rate", "{:.1%}", False),
                                                ("Revenue", "revenue", "{:,.0f}", False)]):
        d = r_cur[key] - r_base[key]
        col.metric(lab, fmt.format(r_cur[key]), (f"{d:+,.0f}" if "{:,.0f}" == fmt else f"{d * 100:+.1f} pp") + " vs default policy",
                   delta_color="inverse" if inv else "normal")
    st.caption("Delta is against the default-parameter policy with the same forecast and no promotion.")
    d1, d2 = r_cur["daily"], r_base["daily"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d1.day, y=d1.on_hand_units, name="On-hand (scenario)", line=dict(color="#1a73e8")))
    fig.add_trace(go.Scatter(x=d2.day, y=d2.on_hand_units, name="On-hand (default)", line=dict(color="#9aa0a6", dash="dot")))
    fig.add_trace(go.Bar(x=d1.day, y=d1.lost_margin, name="Lost margin (scenario)", marker_color="rgba(217,48,37,0.6)", yaxis="y2"))
    fig.update_layout(height=380, yaxis_title="units", yaxis2=dict(overlaying="y", side="right", title="lost margin"),
                      xaxis_title="day", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(pd.DataFrame({"scenario": r_cur, "default": r_base}).drop(index="daily").T
                 .astype(float).style.format("{:,.2f}"), width="stretch")

# ============================================================================ Explainability
with tabs[4]:
    a, b = st.columns(2)
    if "shap_xgb" in res:
        sx = res["shap_xgb"]
        a.subheader("XGBoost – TreeSHAP")
        fig = px.bar(sx["importance"].head(15).iloc[::-1], x="mean_abs_shap", y="feature", color="group", orientation="h",
                     title="Global variable importance (mean |SHAP|)")
        a.plotly_chart(fig, width="stretch")
        a.plotly_chart(px.bar(sx["groups"], x="group", y="mean_abs_shap", title="Static metadata vs temporal vs promo"),
                       width="stretch")
    else:
        a.warning(res.get("shap_xgb_error", "SHAP for XGBoost unavailable"))
    if "shap_tft" in res:
        st_ = res["shap_tft"]
        b.subheader("Contract-Sparse Transformer – group Kernel-SHAP")
        fig = px.bar(st_["feature_groups"].iloc[::-1], x="mean_abs_shap", y="group", orientation="h",
                     title="Global importance by feature group (scaled P50 units)")
        b.plotly_chart(fig, width="stretch")
        b.plotly_chart(px.bar(st_["temporal_blocks"], x="block", y="mean_shap", title="Temporal attribution (signed mean SHAP)"),
                       width="stretch")
    else:
        b.warning(res.get("shap_tft_error", "SHAP for the Transformer unavailable"))
    if "shap_tft" in res:
        st.subheader("Single series explanation")
        ps = res["shap_tft"]["per_series_feature"]
        sid = st.selectbox("Series", [res["series_ids"][i].replace("__", " / ") for i in ps.series_idx])
        row = ps.iloc[[res["series_ids"][i].replace("__", " / ") for i in ps.series_idx].index(sid)].drop("series_idx")
        fig = go.Figure(go.Bar(x=row.values, y=row.index, orientation="h",
                               marker_color=["#188038" if v > 0 else "#d93025" for v in row.values]))
        fig.update_layout(title="Contribution to the mean P50 forecast vs the average SKU (scaled units)", height=380)
        st.plotly_chart(fig, width="stretch")

# ============================================================================ Anomalies & promotions
with tabs[5]:
    an = res["anomalies"]
    st.subheader("Demand anomalies")
    causes = st.multiselect("Cause", sorted(an.cause.unique()), default=[c for c in an.cause.unique() if c.startswith("Unexplained")])
    sel = an[an.cause.isin(causes)]
    if store_f != "All":
        sel = sel[sel.series_id.str.startswith(store_f)]
    st.write(f"{len(sel)} flagged points ({len(an)} in total; promotions and holidays are expected spikes).")
    st.dataframe(sel.drop(columns=["series_idx", "t"]).sort_values("z_score", key=np.abs, ascending=False).head(200),
                 width="stretch", hide_index=True)
    sids = sorted(sel.series_id.unique()) or [res["series_ids"][0]]
    sid = st.selectbox("Plot series", sids)
    s_i = res["series_ids"].index(sid)
    dts = pd.to_datetime(res["dates"])
    fig = go.Figure(go.Scatter(x=dts, y=res["sales"][s_i], name="Sales", line=dict(color="#5f6368", width=1)))
    pts = an[an.series_id == sid]
    for cause, colr in [("Holiday", "#188038"), ("Promotion", "#f29900"), ("Unexplained (viral trend / data error?)", "#d93025")]:
        pp = pts[pts.cause == cause]
        fig.add_trace(go.Scatter(x=pp.date, y=pp.sales, mode="markers", name=cause, marker=dict(color=colr, size=9)))
    fig.update_layout(height=380, title=f"Anomaly flags – {sid.replace('__', ' / ')}")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Promotion sensitivity (non-linear)")
    pr = res["promo"]
    if pr.get("ok"):
        c = st.columns(4)
        c[0].metric("Max uplift (Umax)", f"+{pr['umax']:.0%}")
        c[1].metric("Half-saturation discount (k)", f"{pr['k']:.0%}")
        c[2].metric("Steepness (h)", f"{pr['h']:.2f}")
        c[3].metric("R² on daily points", f"{pr['r2']:.2f}", help="Daily sales are very noisy, so R² on raw points is low even when the curve is right.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["points"].discount, y=(pr["points"].ratio - 1) * 100, mode="markers", name="Promo days",
                                 marker=dict(size=4, color="rgba(95,99,104,0.35)")))
        fig.add_trace(go.Scatter(x=pr["curve"].discount, y=pr["curve"].uplift_pct, name="Hill fit", line=dict(color="#1a73e8", width=3)))
        fig.update_layout(xaxis_title="discount depth", yaxis_title="uplift vs baseline (%)", height=380, xaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Not enough promotion days to fit an uplift curve.")

# ============================================================================ Model
with tabs[6]:
    st.subheader("Training curves")
    cols = st.columns(2)
    for col, name in zip(cols, ["TFT-Sparse", "LSTM"]):
        h = pd.DataFrame(res["history"][name])
        fig = px.line(h, x="epoch", y=["train_loss", "val_loss"], title=name)
        col.plotly_chart(fig, width="stretch")
    h = pd.DataFrame(res["history"]["TFT-Sparse"])
    st.plotly_chart(px.line(h, x="epoch", y=["mech_loss", "tau"], title="Mechanism loss and Gumbel temperature (annealed)"),
                    width="stretch")
    st.subheader("Configuration")
    st.json({k: v for k, v in res["cfg"].items()})
    st.caption(f"Pipeline runtime {res['runtime_sec']:.0f}s on {res['device']}")
