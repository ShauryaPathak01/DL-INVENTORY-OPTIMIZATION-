"""
inventory.py - probabilistic inventory recommendations, risk / expected cost and a what-if simulator.

Policy: periodic-review (s, S)
------------------------------
Inputs (InventoryParams): lead time L, review period P, target service level, holding / stock-out costs.

Demand model over a window of n days built from the quantile forecast:
    daily sigma_d  = (P90 - P10) / (2 * 1.2816)              (Gaussian-equivalent spread)
    mu_n           = sum of P50 over the window              (point forecast)
    sigma_n^2      = sum sigma_d^2 + rho * ((sum sigma_d)^2 - sum sigma_d^2)   (rho = error autocorrelation)
Outputs
    Safety Stock        SS  = z(SL) * sigma_L
    Reorder Point       ROP = mu_L + SS                              (lead-time window)
    Order-up-to level   OUT = mu_(L+P) + z(SL) * sigma_(L+P)         (protection interval = lead time + review)
    Inventory position  IP  = on_hand + on_order
    Suggested Order Qty SOQ = max(0, OUT - IP)  if IP <= ROP else 0  (rounded up to pack size / min order)
Risk (Normal approximation)
    Stock-out risk   = P(D_L > IP)
    Overstock risk   = P(leftover after L+P days > overstock_threshold * mu_(L+P))
    Expected cost    = holding * E[leftover] * days + stock-out penalty * E[shortage]  (normal loss functions)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.stats import norm

Z90 = 1.2815515655446004


@dataclass
class InventoryParams:
    lead_time: int = 3
    review_period: int = 1
    service_level: float = 0.95
    rho: float = 0.2
    holding_rate_per_day: float = 0.002
    cost_ratio: float = 0.7
    goodwill_mult: float = 0.5
    order_cost: float = 1.0
    overstock_threshold: float = 0.5
    pack_size: int = 1
    min_order_qty: int = 0

    @classmethod
    def from_config(cls, cfg):
        keys = ["lead_time", "review_period", "service_level", "rho", "holding_rate_per_day", "cost_ratio",
                "goodwill_mult", "order_cost", "overstock_threshold"]
        return cls(**{k: getattr(cfg, k) for k in keys})

    def with_(self, **kw):
        return replace(self, **kw)


# ----------------------------------------------------------------------------- helpers
def interval_from_point(point, resid_sigma):
    """Turn a point forecast into (q10, q50, q90) with a per-series residual sigma (for baseline models)."""
    sig = np.asarray(resid_sigma)[:, None]
    return np.clip(point - Z90 * sig, 0, None), point, point + Z90 * sig


def window_stats(q10, q50, q90, start, length, rho=0.2):
    """mu, sigma of total demand over days [start, start+length). Beyond the forecast horizon the last
    7 forecast days are repeated cyclically."""
    N, H = q50.shape
    idx = np.arange(start, start + length)
    per = min(7, H)
    idx = np.where(idx < H, idx, H - per + ((idx - H) % per))
    sd = (q90 - q10)[:, idx] / (2 * Z90)
    mu = q50[:, idx].sum(1)
    var = (sd ** 2).sum(1) + rho * (sd.sum(1) ** 2 - (sd ** 2).sum(1))
    return mu, np.sqrt(np.maximum(var, 1e-12))


def _round_order(q, params):
    q = np.where(q > 0, np.maximum(q, params.min_order_qty), 0)
    return np.ceil(q / params.pack_size) * params.pack_size


def _loss_terms(Q, mu, sigma):
    """Normal loss functions: expected leftover and expected shortage for stock level Q."""
    z = (Q - mu) / sigma
    over = sigma * (z * norm.cdf(z) + norm.pdf(z))
    short = sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
    return over, short


# ----------------------------------------------------------------------------- recommendations
def compute_schedule(meta, q10, q50, q90, on_hand, on_order, price, params: InventoryParams, point=None):
    """Per-SKU/store inventory schedule + risk + expected cost. All arrays are [N] or [N,H]."""
    L, P, z = params.lead_time, params.review_period, norm.ppf(params.service_level)
    base = q50 if point is None else point
    mu_L, sig_L = window_stats(q10, base, q90, 0, L, params.rho)
    mu_R, sig_R = window_stats(q10, base, q90, 0, L + P, params.rho)
    ss = z * sig_L
    rop = mu_L + ss
    out = mu_R + z * sig_R
    ip = on_hand + on_order
    flag = ip <= rop
    soq = _round_order(np.where(flag, np.maximum(out - ip, 0), 0), params)

    cost = params.cost_ratio * price
    h = params.holding_rate_per_day * cost
    pen = (price - cost) * (1 + params.goodwill_mult)
    Q = ip + soq
    stockout_risk = 1 - norm.cdf((ip - mu_L) / sig_L)
    overstock_risk = norm.cdf((Q - params.overstock_threshold * mu_R - mu_R) / sig_R)
    over, short = _loss_terms(Q, mu_R, sig_R)
    hold_cost, short_cost = h * over * (L + P), pen * short
    df = meta[["store_id", "item_id", "dept_id"]].copy().reset_index(drop=True)
    df["forecast_P10_LT"] = np.round(window_stats(q10, q10, q10, 0, L, 0)[0], 2)
    df["forecast_P50_LT"] = np.round(mu_L, 2)
    df["forecast_P90_LT"] = np.round(window_stats(q90, q90, q90, 0, L, 0)[0], 2)
    df["safety_stock_SS"] = np.round(ss, 2)
    df["reorder_point_ROP"] = np.round(rop, 2)
    df["order_up_to"] = np.round(out, 2)
    df["on_hand"], df["on_order"] = np.round(on_hand, 2), np.round(on_order, 2)
    df["inventory_position"] = np.round(ip, 2)
    df["reorder_now"] = flag
    df["suggested_order_qty_SOQ"] = soq
    df["stockout_risk"] = np.round(stockout_risk, 4)
    df["overstock_risk"] = np.round(overstock_risk, 4)
    df["expected_holding_cost"] = np.round(hold_cost, 3)
    df["expected_stockout_cost"] = np.round(short_cost, 3)
    df["expected_cost"] = np.round(hold_cost + short_cost, 3)
    df["expected_revenue"] = np.round(price * (mu_R - short), 2)
    return df


# ----------------------------------------------------------------------------- what-if simulation
def simulate_policy(demand, q10, q50, q90, price, init_on_hand, params: InventoryParams,
                    promo_uplift=0.0, promo_window=(0, 0), forecast_aware=True):
    """Day-by-day simulation of the (s,S) policy driven by the forecast (re-read every review).
    demand/q* are [N,T]. Promo scenario multiplies true demand by (1+uplift) inside ``promo_window``;
    if ``forecast_aware`` the forecast is scaled too (planned promo) else the policy is blind to it."""
    N, T = demand.shape
    demand = demand.astype(float).copy()
    q10, q50, q90 = (a.astype(float).copy() for a in (q10, q50, q90))
    a, b = promo_window
    if promo_uplift != 0 and b > a:
        demand[:, a:b] *= 1 + promo_uplift
        if forecast_aware:
            for arr in (q10, q50, q90):
                arr[:, a:b] *= 1 + promo_uplift
    L, P, z = max(1, params.lead_time), max(1, params.review_period), norm.ppf(params.service_level)
    cost = params.cost_ratio * price
    h = params.holding_rate_per_day * cost
    pen = (price - cost) * (1 + params.goodwill_mult)
    on_hand = init_on_hand.astype(float).copy()
    pipe = np.zeros((N, L))
    sold_t, lost_t, oh_t = np.zeros(T), np.zeros(T), np.zeros(T)
    n_orders = np.zeros(N)
    stock_days = np.zeros(N)
    hold_units = np.zeros(N)
    for t in range(T):
        on_hand += pipe[:, 0]
        pipe = np.concatenate([pipe[:, 1:], np.zeros((N, 1))], 1)
        d = demand[:, t]
        s = np.minimum(on_hand, d)
        lost = d - s
        on_hand -= s
        sold_t[t], lost_t[t], oh_t[t] = (s * price).sum(), (lost * (price - cost)).sum(), on_hand.sum()
        stock_days += (lost > 1e-9)
        hold_units += on_hand
        if t % P == 0:
            ip = on_hand + pipe.sum(1)
            mu_L, sg_L = window_stats(q10, q50, q90, t + 1, L, params.rho)
            mu_R, sg_R = window_stats(q10, q50, q90, t + 1, L + P, params.rho)
            rop = mu_L + z * sg_L
            out = mu_R + z * sg_R
            order = _round_order(np.where(ip <= rop, np.maximum(out - ip, 0), 0), params)
            pipe[:, L - 1] += order
            n_orders += order > 0
    return _finalize(demand, price, cost, h, pen, params, hold_units, n_orders, sold_t, lost_t, oh_t, stock_days)


def _finalize(demand, price, cost, h, pen, params, hold_units, n_orders, sold_rev_t, lost_margin_t, oh_t, stock_days):
    N, T = demand.shape
    revenue = float(sold_rev_t.sum())
    potential = float((demand * price[:, None]).sum())
    # lost margin already includes (price-cost); add goodwill part via pen/(price-cost) factor
    lost_pen = float((lost_margin_t.sum()) * (1 + params.goodwill_mult))
    holding = float((hold_units * h).sum())
    ordering = float(n_orders.sum() * params.order_cost)
    demand_days = float((demand > 0).sum())
    return {
        "service_level": float(1 - stock_days.sum() / max(demand_days, 1)),     # in-stock rate (SKU-days)
        "fill_rate": float(revenue / max(potential, 1e-9)),                      # revenue-weighted fill rate
        "inventory_cost": holding + lost_pen + ordering,
        "holding_cost": holding, "stockout_cost": lost_pen, "ordering_cost": ordering,
        "revenue": revenue, "lost_revenue": potential - revenue,
        "avg_inventory_units": float(oh_t.mean()),
        "daily": pd.DataFrame({"day": np.arange(T), "on_hand_units": oh_t, "revenue": sold_rev_t,
                               "lost_margin": lost_margin_t}),
    }
