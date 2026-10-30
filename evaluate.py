"""
evaluate.py - metrics (pinball, P10-P90 coverage, MAE/RMSE/WAPE, WRMSSE) and hierarchical reconciliation.

Hierarchy (bottom level = store x item):
    Total -> Store -> Dept -> Store x Dept -> Store x Item
``build_summing_matrix`` returns the summing matrix S [N_all, N_bottom]. Reconciliation options:
    * bottom-up  (default in the pipeline): aggregate bottom-level forecasts through S (coherent by construction);
                 ``aggregate_quantiles`` does this for P10/P50/P90 using a Gaussian / independence approximation.
    * MinT-shrink (``mint_reconcile``): optimal combination of base forecasts at ALL nodes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.2815515655446004      # N(0,1) 90th percentile


# ----------------------------------------------------------------------------- point / probabilistic metrics
def pinball_np(y, q_preds, quantiles=(0.1, 0.5, 0.9)):
    """y [N,T], q_preds [N,T,Q] -> mean pinball loss."""
    q = np.asarray(quantiles)[None, None, :]
    e = y[..., None] - q_preds
    return float(np.maximum(q * e, (q - 1) * e).mean())


def interval_coverage(y, lo, hi):
    """Empirical share of actuals inside [lo, hi] (nominal 80 % for P10-P90)."""
    return float(((y >= lo) & (y <= hi)).mean())


def mae(y, p): return float(np.abs(y - p).mean())
def rmse(y, p): return float(np.sqrt(((y - p) ** 2).mean()))
def wape(y, p): return float(np.abs(y - p).sum() / max(np.abs(y).sum(), 1e-9))
def mase(y_train, y_true, y_pred, season=7):
    """Mean Absolute Scaled Error relative to seasonal naive on training data."""
    scale = np.mean(np.abs(y_train[:, season:] - y_train[:, :-season])) + 1e-6
    return float(np.mean(np.abs(y_true - y_pred)) / scale)



# ----------------------------------------------------------------------------- hierarchy
def build_summing_matrix(meta: pd.DataFrame):
    """Returns S [N_all,N_bottom], node labels, node level index, level names."""
    N = len(meta)
    store, dept = meta["store_id"].astype(str).values, meta["dept_id"].astype(str).values
    levels = [("Total", np.array(["Total"] * N)),
              ("Store", store),
              ("Dept", dept),
              ("Store x Dept", np.char.add(np.char.add(store, " / "), dept)),
              ("Store x Item", meta["store_id"].astype(str).values + " / " + meta["item_id"].astype(str).values)]
    blocks, labels, lvl = [], [], []
    for li, (_, keys) in enumerate(levels):
        uniq, inv = np.unique(keys, return_inverse=True)
        M = np.zeros((len(uniq), N))
        M[inv, np.arange(N)] = 1
        blocks.append(M)
        labels += [str(u) for u in uniq]
        lvl += [li] * len(uniq)
    return np.vstack(blocks), labels, np.array(lvl), [n for n, _ in levels]


def aggregate_quantiles(S, q10, q50, q90):
    """Bottom-up probabilistic aggregation: means add; variances add (independence) -> Gaussian quantiles.
    Summing P90s would over-state the aggregate P90, hence this approximation. Returns three [N_all,H] arrays."""
    sigma = (q90 - q10) / (2 * Z90)
    mu = S @ q50
    sd = np.sqrt(S @ (sigma ** 2))
    return np.clip(mu - Z90 * sd, 0, None), mu, mu + Z90 * sd


def mint_reconcile(S, base, residuals):
    """MinT-shrink. base [N_all,H] base forecasts for every node, residuals [N_all,R] in-sample errors of those
    base forecasts. Returns reconciled forecasts for ALL nodes (coherent): S (S'W^-1 S)^-1 S'W^-1 base."""
    from sklearn.covariance import LedoitWolf
    W = LedoitWolf().fit(residuals.T).covariance_
    Wi = np.linalg.pinv(W)
    P = np.linalg.pinv(S.T @ Wi @ S) @ S.T @ Wi
    return S @ (P @ base)


# ----------------------------------------------------------------------------- WRMSSE (M5 metric)
def wrmsse(S, level_of_row, y_train, y_true, y_pred, revenue_weights):
    """Weighted RMSSE across the whole hierarchy (each level has total weight 1/n_levels,
    inside a level weights are revenue shares). y_* are bottom-level [N,*]."""
    A_tr, A_true, A_pred = S @ y_train, S @ y_true, S @ y_pred
    w_rev = S @ revenue_weights
    scores, weights = np.zeros(len(S)), np.zeros(len(S))
    n_lv = len(np.unique(level_of_row))
    for j in range(len(S)):
        h = A_tr[j]
        nz = np.nonzero(h)[0]
        h = h[nz[0]:] if len(nz) else h
        d = np.mean(np.diff(h) ** 2) if len(h) > 1 else 0.0
        if d <= 0:
            continue
        scores[j] = np.sqrt(np.mean((A_true[j] - A_pred[j]) ** 2) / d)
        weights[j] = 1.0
    for li in np.unique(level_of_row):
        idx = np.where(level_of_row == li)[0]
        tot = w_rev[idx].sum()
        weights[idx] = (w_rev[idx] / tot if tot > 0 else 1.0 / len(idx)) * weights[idx] / n_lv
    return float((scores * weights).sum() / max(weights.sum(), 1e-9) * 1.0)
