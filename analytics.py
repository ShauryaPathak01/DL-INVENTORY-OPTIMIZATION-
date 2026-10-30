"""
analytics.py - explainability (SHAP), anomaly detection and non-linear promotion analysis.

SHAP
----
* XGBoost: exact ``shap.TreeExplainer`` -> global importance, static store/SKU metadata impact, temporal features.
* Contract-Sparse Transformer: attention models are not supported by Deep/Tree explainers, so we use
  ``shap.KernelExplainer`` over *feature groups* (a group is either kept or replaced by a background value).
  With <= 11 groups the Shapley values are computed by full enumeration (exact). Two groupings:
    - feature groups   (history sales, calendar, promo, price, store id, item id, ...)
    - temporal blocks  (look-back windows -28..-1 days and the known-future covariates)
  Explained scalar = mean P50 forecast over the horizon (scaled units).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import torch

from data import SEQ_FEATURE_NAMES, tab_feature_names

try:
    import shap
except Exception:            # pragma: no cover
    shap = None


# ============================================================================ SHAP - XGBoost
def _tab_group(name: str) -> str:
    if name in ("store_c", "item_c", "dept_c", "cat_c", "scale"):
        return "Static store / SKU metadata"
    if name in ("dow", "dom", "month", "woy", "holiday"):
        return "Calendar & holidays"
    if name in ("promo", "price_rel"):
        return "Promotion & price"
    return "Lags & rolling statistics"


def xgb_shap(xgb_model, X: pd.DataFrame, max_rows=1500, seed=0):
    names = xgb_model.features
    Xs = X[names].sample(n=min(max_rows, len(X)), random_state=seed)
    sv = shap.TreeExplainer(xgb_model.model).shap_values(Xs)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv).mean(0)})
    imp["group"] = imp["feature"].map(_tab_group)
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    grp = imp.groupby("group")["mean_abs_shap"].sum().sort_values(ascending=False).reset_index()
    return {"importance": imp, "groups": grp, "values": sv, "X": Xs}


# ============================================================================ SHAP - Transformer (group Kernel SHAP)
def _exact_shapley(f, G):
    """Fallback exact Shapley values (used if the shap package is missing). f maps a [m,G] 0/1 matrix to [m]."""
    vals = np.zeros(G)
    allz = np.array(list(itertools.product([0, 1], repeat=G)), dtype=float)
    fv = dict(zip(map(tuple, allz.astype(int)), f(allz)))
    for i in range(G):
        for z in itertools.product([0, 1], repeat=G):
            if z[i] == 1:
                continue
            k = sum(z)
            w = math.factorial(k) * math.factorial(G - k - 1) / math.factorial(G)
            z2 = list(z); z2[i] = 1
            vals[i] += w * (fv[tuple(z2)] - fv[z])
    return vals


def _seq_groups(L, H, n_feat):
    """Feature-group masks over the [L+H, F] sequence tensor + which static inputs belong to the group."""
    ix = {n: i for i, n in enumerate(SEQ_FEATURE_NAMES)}
    def m(cols, rows=slice(None)):
        a = np.zeros((L + H, n_feat), bool)
        a[rows, cols] = True
        return a
    cal = [ix[c] for c in ["dow_sin", "dow_cos", "woy_sin", "woy_cos", "month_sin", "month_cos"]]
    groups = [
        ("History: sales", m([ix["obs_sales"]]), None),
        ("History: 7-step rolling mean", m([ix["roll7"]]), None),
        ("Calendar", m(cal), None),
        ("Promotion flag", m([ix["promo"]]), None),
        ("Holiday flag", m([ix["holiday"]]), None),
        ("Price (relative)", m([ix["price_rel"]]), None),
        ("Static: store id", None, ("cat", [0])),
        ("Static: item id", None, ("cat", [1])),
        ("Static: dept / category", None, ("cat", [2, 3])),
        ("Static: SKU scale & zero-share", None, ("num", None)),
    ]
    return groups


def _time_groups(L, H, n_feat):
    blocks = []
    for a, b in [(0, 7), (7, 14), (14, 21), (21, 28)]:
        if b > L:
            b = L
        if a >= b:
            continue
        m = np.zeros((L + H, n_feat), bool)
        m[L - b:L - a, :] = True
        blocks.append((f"Look-back t-{b}..t-{a + 1}", m, None))
    m = np.zeros((L + H, n_feat), bool)
    m[L:, :] = True
    blocks.append(("Known future covariates (horizon)", m, None))
    return blocks


def _run_group_shap(model, X, SC, SN, base_x, base_num, groups, device, nsamples):
    G = len(groups)
    out = np.zeros((len(X), G))
    model.eval()
    for j in range(len(X)):
        def f(Z):
            Z = np.asarray(Z)
            m = Z.shape[0]
            xs = np.repeat(X[j][None], m, 0)
            sc = np.repeat(SC[j][None], m, 0)
            sn = np.repeat(SN[j][None], m, 0)
            for g, (_, mask, stat) in enumerate(groups):
                off = Z[:, g] < 0.5
                if mask is not None:
                    sel = off[:, None, None] & mask[None]
                    xs = np.where(sel, base_x[None], xs)
                elif stat[0] == "cat":
                    for c in stat[1]:
                        sc[off, c] = 0                       # id 0 = "unknown" embedding
                else:
                    sn[off] = base_num
            with torch.no_grad():
                p, _ = model(torch.from_numpy(xs.astype(np.float32)).to(device),
                             torch.from_numpy(sc).to(device), torch.from_numpy(sn.astype(np.float32)).to(device))
            return p[..., 1].mean(1).cpu().numpy()
        if shap is not None:
            ex = shap.KernelExplainer(f, np.zeros((1, G)))
            out[j] = np.asarray(ex.shap_values(np.ones((1, G)), nsamples=nsamples, silent=True)).reshape(-1)[:G]
        else:
            out[j] = _exact_shapley(f, G)
    return out


def tft_group_shap(model, sampler_train, sampler_explain, device, n_background=400, nsamples=2500, max_samples=40):
    """Group SHAP for the transformer. Returns feature-group and temporal-block attributions."""
    rng = np.random.default_rng(0)
    bg_sel = rng.choice(len(sampler_train.s), size=min(n_background, len(sampler_train.s)), replace=False)
    bx, bsc, bsn, _ = sampler_train.numpy_batch(bg_sel)
    base_x, base_num = bx.mean(0), bsn.mean(0)
    n = len(sampler_explain.s)
    sel = np.sort(rng.choice(n, size=min(max_samples, n), replace=False))
    X, SC, SN, _ = sampler_explain.numpy_batch(sel)
    b = sampler_explain.b
    L, H, F = b.L, b.H, X.shape[-1]
    fg, tg = _seq_groups(L, H, F), _time_groups(L, H, F)
    fv = _run_group_shap(model, X, SC, SN, base_x, base_num, fg, device, nsamples)
    tv = _run_group_shap(model, X, SC, SN, base_x, base_num, tg, device, nsamples)
    series = sampler_explain.s[sel]
    return {
        "feature_groups": pd.DataFrame({"group": [g[0] for g in fg], "mean_abs_shap": np.abs(fv).mean(0)})
                            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True),
        "temporal_blocks": pd.DataFrame({"block": [g[0] for g in tg], "mean_abs_shap": np.abs(tv).mean(0),
                                         "mean_shap": tv.mean(0)}),
        "per_series_feature": pd.DataFrame(fv, columns=[g[0] for g in fg]).assign(series_idx=series),
        "per_series_temporal": pd.DataFrame(tv, columns=[g[0] for g in tg]).assign(series_idx=series),
    }


# ============================================================================ anomaly detection
def detect_anomalies(b, z_thresh=3.5, window=56):
    """Seasonal-robust anomaly detection on the history.
    expected = median of the same weekday over the previous 4 seasons; residual scaled by max(1.4826*rolling MAD,
    Poisson sd). Cause is attributed to holidays (+-1 day), promotions, or left 'unexplained' (viral trend / data error)."""
    S, T = b.sales.shape
    sea = b.season
    lags = []
    for k in range(1, 5):
        a = np.full((S, T), np.nan)
        if k * sea < T:
            a[:, k * sea:] = b.sales[:, :T - k * sea]
        lags.append(a)
    with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        expected = np.nanmedian(np.stack(lags), axis=0)
    res = b.sales - expected
    mad = pd.DataFrame(np.abs(res).T).rolling(window, min_periods=14).median().values.T
    denom = np.maximum(1.4826 * mad, np.sqrt(np.maximum(expected, 0) + 1.0))
    z = res / denom
    z = np.where(np.isnan(z), 0, z)
    rows = []
    hol = b.holiday
    for s, t in np.argwhere(np.abs(z) > z_thresh):
        near_hol = hol[s, max(0, t - 1):t + 2].max() > 0
        cause = "Holiday" if near_hol else ("Promotion" if b.promo[s, t] > 0 else "Unexplained (viral trend / data error?)")
        rows.append({"series_idx": s, "series_id": b.series_ids[s], "date": b.dates[t], "t": t,
                     "sales": float(b.sales[s, t]), "expected": float(expected[s, t]), "z_score": float(z[s, t]),
                     "type": "spike" if z[s, t] > 0 else "drop", "cause": cause})
    cols = ["series_idx", "series_id", "date", "t", "sales", "expected", "z_score", "type", "cause"]
    return pd.DataFrame(rows, columns=cols)


# ============================================================================ promotion analysis
def _hill(d, umax, k, h):
    return umax * d ** h / (d ** h + k ** h)


def promotion_analysis(b):
    """Non-linear promotion sensitivity: ratio of promo-day sales to a non-promo rolling-median baseline vs
    discount depth, fitted with a saturating Hill curve  uplift(d) = Umax * d^h / (d^h + k^h)."""
    from scipy.optimize import curve_fit
    sales, promo = b.sales.astype(float), b.promo
    base = pd.DataFrame(np.where(promo > 0, np.nan, sales).T).rolling(28, min_periods=7).median().values.T
    # weekday adjustment: per-series median ratio of non-promo sales to the rolling baseline, by day of week
    dow = np.asarray(b.dates.dayofweek)
    ratio0 = np.where((promo == 0) & (base >= 1), sales / np.maximum(base, 1e-9), np.nan)
    dfac = np.ones((sales.shape[0], 7))
    for k in range(7):
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore")
            v = np.nanmedian(ratio0[:, dow == k], axis=1)
        dfac[:, k] = np.where(np.isnan(v), 1.0, v)
    base = base * dfac[:, dow]
    depth = np.clip(1 - b.price_rel, 0, 0.8)
    ok = (promo > 0) & (depth > 0.01) & (base >= 1) & (b.holiday == 0)
    d, r = depth[ok], sales[ok] / base[ok]
    if len(d) < 30:
        return {"ok": False}
    u = np.clip(r - 1, -0.5, 10)
    try:
        (umax, k, h), _ = curve_fit(_hill, d, u, p0=[1.0, 0.2, 2.0], bounds=([0, 0.01, 0.5], [10, 1.0, 6.0]),
                                loss="soft_l1", f_scale=0.5, max_nfev=5000)
        pred = _hill(d, umax, k, h)
        r2 = float(1 - ((u - pred) ** 2).sum() / max(((u - u.mean()) ** 2).sum(), 1e-9))
    except Exception:
        umax, k, h, r2 = float(np.nan), float(np.nan), float(np.nan), float(np.nan)
    grid = np.linspace(0, max(0.5, d.max()), 60)
    curve = pd.DataFrame({"discount": grid, "uplift_pct": 100 * _hill(grid, umax, k, h)})
    bins = pd.cut(d, [0, .05, .1, .15, .2, .25, .3, .4, .8])
    binned = pd.DataFrame({"bin": bins, "ratio": r}).groupby("bin", observed=True)["ratio"].agg(["median", "count"]).reset_index()
    binned["bin"] = binned["bin"].astype(str)
    sample = pd.DataFrame({"discount": d, "ratio": r}).sample(n=min(1500, len(d)), random_state=0)
    return {"ok": True, "umax": float(umax), "k": float(k), "h": float(h), "r2": r2, "curve": curve,
            "binned": binned, "points": sample}


def uplift_at(depth, res):
    return float(_hill(np.asarray(depth), res["umax"], res["k"], res["h"]))
