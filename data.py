"""
data.py - loaders, dummy-data generator, preprocessing, feature engineering and
leakage-free time-based windowing.

Canonical long schema produced by every loader
----------------------------------------------
date, store_id, item_id, dept_id, cat_id, sales, sell_price, promo, holiday

Time-based split (no leakage)
-----------------------------
With T observed steps and horizon H:

    |------------- train -------------|-- val (2H) --|-- test (2H) --|-- future (H) -->
    0                              n_train        n_val_end          T

* A training window with origin ``o`` only uses history ``[o-L, o)`` and targets
  ``[o, o+H)`` with ``o + H <= n_train``  ->  no target of any train window lies
  in val/test.
* Val / test are evaluated as two non-overlapping H-blocks (origins ``start`` and
  ``start+H``); the second block sees the actuals of the first, exactly as it
  would in production.
* Scaling constants, price references and encoders are fitted on ``[0, n_train)``.
* Lag / rolling features for the tabular (XGBoost) model use a shift >= H, so every
  feature of every horizon step is known at the forecast origin.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import Config

N_KNOWN = 9            # dow(2) + week(2) + month(2) + promo + holiday + price_rel
N_SEQ_FEATURES = 12    # obs_sales, roll7, is_future + N_KNOWN
SEQ_FEATURE_NAMES = ["obs_sales", "roll7", "is_future",
                     "dow_sin", "dow_cos", "woy_sin", "woy_cos", "month_sin", "month_cos",
                     "promo", "holiday", "price_rel"]
SCHEMA = ["date", "store_id", "item_id", "dept_id", "cat_id", "sales", "sell_price", "promo", "holiday"]


# ----------------------------------------------------------------------------
# Holidays (rule based fallback, used for dummy data and for future dates)
# ----------------------------------------------------------------------------
def holiday_flag(dates) -> np.ndarray:
    d = pd.DatetimeIndex(dates)
    fixed = (((d.month == 1) & (d.day == 1)) | ((d.month == 7) & (d.day == 4)) |
             ((d.month == 12) & (d.day == 25)) | ((d.month == 12) & (d.day == 24)) |
             ((d.month == 11) & (d.dayofweek == 3) & (d.day >= 22) & (d.day <= 28)))
    return np.asarray(fixed).astype(int)


# ----------------------------------------------------------------------------
# Dummy data generator
# ----------------------------------------------------------------------------
def generate_dummy_data(n_stores=3, n_items=10, n_days=730, start="2023-01-01", seed=0) -> pd.DataFrame:
    """Synthetic multi-store retail data with weekly/yearly seasonality, trend, holidays,
    saturating (Hill-shaped) promotion uplift and injected anomalies."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days)
    depts = ["FOODS_1", "FOODS_2", "HOBBIES_1", "HOUSEHOLD_1"]
    cats = {"FOODS_1": "FOODS", "FOODS_2": "FOODS", "HOBBIES_1": "HOBBIES", "HOUSEHOLD_1": "HOUSEHOLD"}
    hol = holiday_flag(dates)
    dow, doy = dates.dayofweek.values, dates.dayofyear.values
    t = np.arange(n_days)
    frames = []
    for s in range(n_stores):
        store_mult = rng.uniform(0.7, 1.5)
        for i in range(n_items):
            dept = depts[i % len(depts)]
            base = (rng.gamma(2.0, 4.0) + 1.0) * store_mult
            weekly = 1 + 0.35 * np.sin(2 * np.pi * (dow - rng.uniform(0, 7)) / 7)
            yearly = 1 + 0.20 * np.sin(2 * np.pi * doy / 365.25 + rng.uniform(0, 6.28))
            trend = 1 + rng.uniform(-0.1, 0.3) * t / n_days
            price0 = rng.uniform(2, 20)
            depth = np.zeros(n_days)
            for _ in range(n_days // 40):                     # promotion blocks
                a = rng.integers(0, n_days - 8)
                depth[a:a + rng.integers(3, 8)] = rng.uniform(0.1, 0.4)
            uplift = 1 + 1.2 * depth ** 2 / (depth ** 2 + 0.2 ** 2)   # Hill saturation
            lam = base * weekly * yearly * trend * uplift * (1 + 0.4 * hol)
            sales = rng.poisson(lam * rng.gamma(20, 1 / 20, size=n_days)).astype(float)
            for _ in range(3):                                # injected anomalies
                k = rng.integers(40, n_days - 1)
                sales[k] = sales[k] * rng.uniform(3, 5) + 5 if rng.random() < 0.6 else 0
            frames.append(pd.DataFrame({
                "date": dates, "store_id": f"ST_{s + 1}", "item_id": f"ITEM_{i + 1:03d}",
                "dept_id": dept, "cat_id": cats[dept], "sales": sales,
                "sell_price": price0 * (1 - depth), "promo": (depth > 0).astype(int), "holiday": hol}))
    return pd.concat(frames, ignore_index=True)[SCHEMA]


# ----------------------------------------------------------------------------
# Real dataset loaders (all return the canonical schema)
# ----------------------------------------------------------------------------
def load_m5(data_dir, max_series=300, seed=0) -> pd.DataFrame:
    """Walmart M5 (primary). Needs sales_train_evaluation.csv (or _validation), calendar.csv,
    sell_prices.csv in ``data_dir``. M5 has no promo flag -> proxy: price < 95 % of the item's median price."""
    d = Path(data_dir)
    f = d / "sales_train_evaluation.csv"
    f = f if f.exists() else d / "sales_train_validation.csv"
    sales = pd.read_csv(f)
    if len(sales) > max_series:
        sales = sales.sample(n=max_series, random_state=seed)
    cal = pd.read_csv(d / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(d / "sell_prices.csv")
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    cal = cal[["d", "date", "wm_yr_wk", "event_name_1"]]
    long = long.merge(cal, on="d", how="left").merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    long["holiday"] = long["event_name_1"].notna().astype(int)
    med = long.groupby("id")["sell_price"].transform("median")
    long["promo"] = (long["sell_price"] < 0.95 * med).astype(int)
    return long[SCHEMA]


def load_walmart_recruiting(data_dir, max_series=100, seed=0) -> pd.DataFrame:
    d = Path(data_dir)
    tr = pd.read_csv(d / "train.csv", parse_dates=["Date"])
    ft = pd.read_csv(d / "features.csv", parse_dates=["Date"])
    st = pd.read_csv(d / "stores.csv")
    md_cols = [f"MarkDown{i}" for i in range(1, 6)]
    
    # Sub-sample to 100 series so it runs in 1-2 minutes on CPU
    keys = tr[["Store", "Dept"]].drop_duplicates()
    if len(keys) > max_series:
        keys = keys.sample(n=max_series, random_state=seed)
    tr = tr.merge(keys, on=["Store", "Dept"], how="inner")
    
    df = tr.merge(ft[["Store", "Date"] + md_cols], on=["Store", "Date"], how="left").merge(st, on="Store", how="left")
    return pd.DataFrame({
        "date": df["Date"], "store_id": "ST_" + df["Store"].astype(str),
        "item_id": "DEPT_" + df["Dept"].astype(str), "dept_id": "DEPT_" + df["Dept"].astype(str),
        "cat_id": df["Type"], "sales": df["Weekly_Sales"].clip(lower=0), "sell_price": 1.0,
        "promo": (df[md_cols].fillna(0).sum(axis=1) > 0).astype(int),
        "holiday": df["IsHoliday"].astype(int)})[SCHEMA]


def load_favorita(data_dir, max_series=300, seed=0) -> pd.DataFrame:
    """Kaggle 'Favorita Store Sales' (daily). train.csv, stores.csv, holidays_events.csv. Not exercised in the tests."""
    d = Path(data_dir)
    tr = pd.read_csv(d / "train.csv", parse_dates=["date"])
    st = pd.read_csv(d / "stores.csv")
    hol = pd.read_csv(d / "holidays_events.csv", parse_dates=["date"])
    hol = hol[(hol["transferred"] == False)]                       # noqa: E712
    tr = tr.merge(st[["store_nbr", "type"]], on="store_nbr", how="left")
    keys = tr[["store_nbr", "family"]].drop_duplicates()
    keys = keys.sample(n=min(max_series, len(keys)), random_state=seed)
    tr = tr.merge(keys, on=["store_nbr", "family"])
    return pd.DataFrame({
        "date": tr["date"], "store_id": "ST_" + tr["store_nbr"].astype(str), "item_id": tr["family"],
        "dept_id": tr["family"], "cat_id": tr["type"], "sales": tr["sales"].clip(lower=0), "sell_price": 1.0,
        "promo": (tr["onpromotion"] > 0).astype(int),
        "holiday": tr["date"].isin(set(hol["date"])).astype(int)})[SCHEMA]


def load_dataset(cfg: Config) -> pd.DataFrame:
    if cfg.source == "dummy":
        return generate_dummy_data(cfg.dummy_stores, cfg.dummy_items, cfg.dummy_days, seed=cfg.seed)
    if cfg.source == "m5":
        return load_m5(cfg.data_dir, cfg.max_series, cfg.seed)
    if cfg.source == "walmart":
        return load_walmart_recruiting(cfg.data_dir, cfg.max_series, cfg.seed)
    if cfg.source == "favorita":
        return load_favorita(cfg.data_dir, cfg.max_series, cfg.seed)
    raise ValueError(f"unknown source {cfg.source}")


# ----------------------------------------------------------------------------
# Bundle: everything the models need, as dense numpy arrays
# ----------------------------------------------------------------------------
@dataclass
class Bundle:
    series_ids: list
    meta: pd.DataFrame            # one row per series: store_id,item_id,dept_id,cat_id
    dates: pd.DatetimeIndex       # length T
    future_dates: pd.DatetimeIndex  # length H
    sales: np.ndarray             # [S,T] raw, missing -> 0
    price: np.ndarray             # [S,T]
    price_ref: np.ndarray         # [S] median train price
    scale: np.ndarray             # [S] mean train sales (fit on train only)
    scaled: np.ndarray            # [S,T] sales / scale
    roll7: np.ndarray             # [S,T] causal 7-step mean of `scaled`
    known: np.ndarray             # [S,T+H,N_KNOWN] known-in-advance covariates
    static_cat: np.ndarray        # [S,4] store,item,dept,cat ids (0 = unknown)
    cat_cards: list               # cardinalities of the 4 categorical statics
    static_num: np.ndarray        # [S,2] standardised log-scale, zero-fraction
    n_train: int
    n_val_end: int
    T: int
    H: int
    L: int
    season: int

    @property
    def S(self):
        return len(self.series_ids)

    # convenience views on history of the known covariates
    @property
    def promo(self):
        return self.known[:, :self.T, 6]

    @property
    def holiday(self):
        return self.known[:, :self.T, 7]

    @property
    def price_rel(self):
        return self.known[:, :self.T, 8]


def build_bundle(df: pd.DataFrame, cfg: Config) -> Bundle:
    """Preprocess (missing values, scaling, categorical encoding) + known-covariate feature engineering."""
    H, L = cfg.horizon, cfg.encoder_len
    df = df.copy()
    df["series_id"] = df["store_id"].astype(str) + "__" + df["item_id"].astype(str)
    series_ids = sorted(df["series_id"].unique())
    meta = (df.drop_duplicates("series_id").set_index("series_id").loc[series_ids,
            ["store_id", "item_id", "dept_id", "cat_id"]].reset_index())
    full = pd.date_range(df["date"].min(), df["date"].max(), freq=cfg.freq)

    def piv(col, agg):
        p = df.pivot_table(index="series_id", columns="date", values=col, aggfunc=agg)
        return p.reindex(index=series_ids, columns=full)

    # ---- missing values ---------------------------------------------------
    sales = piv("sales", "sum").fillna(0.0).clip(lower=0).values.astype(np.float32)
    price = piv("sell_price", "first").ffill(axis=1).bfill(axis=1).fillna(1.0).values.astype(np.float32)
    promo = piv("promo", "max").fillna(0).values.astype(np.float32)
    hol_vec = piv("holiday", "max").fillna(0).max(axis=0).values.astype(np.float32)   # per date
    S, T = sales.shape
    n_val_end = T - cfg.test_days
    n_train = n_val_end - cfg.val_days
    assert n_train > L + H + 2, "series too short for the requested window / split sizes"

    future_dates = pd.date_range(full[-1], periods=H + 1, freq=cfg.freq)[1:]
    all_dates = full.append(future_dates)

    # ---- scaling fitted on TRAIN only ------------------------------------
    scale = sales[:, :n_train].mean(1)
    scale = np.where(scale < 1e-3, 1.0, scale).astype(np.float32)
    scaled = sales / scale[:, None]
    roll7 = pd.DataFrame(scaled.T).rolling(7, min_periods=1).mean().values.T.astype(np.float32)
    price_ref = np.median(price[:, :n_train], axis=1)
    price_ref = np.where(price_ref <= 0, 1.0, price_ref).astype(np.float32)
    price_rel = price / price_ref[:, None]

    # ---- known covariates (history + future) -----------------------------
    dow = all_dates.dayofweek.values
    woy = np.asarray(all_dates.isocalendar().week).astype(float)
    month = all_dates.month.values
    cal = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                    np.sin(2 * np.pi * woy / 52), np.cos(2 * np.pi * woy / 52),
                    np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12)], 1)
    cal = np.broadcast_to(cal[None], (S, T + H, 6))
    promo_ext = np.concatenate([promo, np.zeros((S, H), np.float32)], 1)        # future promo unknown -> 0
    hol_ext = np.broadcast_to(np.concatenate([hol_vec, holiday_flag(future_dates)])[None], (S, T + H))
    price_rel_ext = np.concatenate([price_rel, np.repeat(price_rel[:, -1:], H, 1)], 1)
    known = np.concatenate([cal, promo_ext[..., None], hol_ext[..., None], price_rel_ext[..., None]], -1).astype(np.float32)

    # ---- categorical statics (fitted on all series, id 0 reserved = unknown) ----
    codes, cards = [], []
    for col in ["store_id", "item_id", "dept_id", "cat_id"]:
        c = pd.Categorical(meta[col]).codes.astype(np.int64) + 1
        codes.append(c)
        cards.append(int(c.max()))
    static_cat = np.stack(codes, 1)
    log_scale = np.log(scale)
    zero_frac = (sales[:, :n_train] == 0).mean(1)
    static_num = np.stack([(log_scale - log_scale.mean()) / (log_scale.std() + 1e-6), zero_frac], 1).astype(np.float32)

    return Bundle(series_ids, meta, full, future_dates, sales, price, price_ref, scale, scaled, roll7, known,
                  static_cat, cards, static_num, n_train, n_val_end, T, H, L, cfg.season)


# ----------------------------------------------------------------------------
# Window sampler for the neural models (fully vectorised, no DataLoader needed)
# ----------------------------------------------------------------------------
class WindowSampler:
    """Yields batches ``x [B,L+H,F]`` = history (observed + known) followed by H future steps that carry only
    known-in-advance covariates (observed channels zeroed + ``is_future`` flag)."""

    def __init__(self, b: Bundle, s_idx, o_idx, batch_size=256, shuffle=False, device="cpu"):
        self.b, self.s, self.o = b, np.asarray(s_idx), np.asarray(o_idx)
        self.bs, self.shuffle, self.device = batch_size, shuffle, device

    def __len__(self):
        return int(np.ceil(len(self.s) / self.bs))

    def numpy_batch(self, sel):
        b, L, H = self.b, self.b.L, self.b.H
        s, o = self.s[sel], self.o[sel]
        B = len(s)
        hist = o[:, None] + np.arange(-L, 0)[None]
        full = o[:, None] + np.arange(-L, H)[None]
        obs = np.zeros((B, L + H), np.float32)
        roll = np.zeros((B, L + H), np.float32)
        isf = np.zeros((B, L + H), np.float32)
        obs[:, :L] = b.scaled[s[:, None], hist]
        roll[:, :L] = b.roll7[s[:, None], hist]
        isf[:, L:] = 1.0
        known = b.known[s[:, None], full]
        x = np.concatenate([obs[..., None], roll[..., None], isf[..., None], known], -1)
        tgt = np.minimum(o[:, None] + np.arange(H)[None], b.T - 1)
        y = b.scaled[s[:, None], tgt]               # meaningless (ignored) for future origins
        return x, b.static_cat[s], b.static_num[s], y

    def _tensors(self, sel):
        x, sc, sn, y = self.numpy_batch(sel)
        d = self.device
        return {"x": torch.from_numpy(x).to(d), "sc": torch.from_numpy(sc).to(d),
                "sn": torch.from_numpy(sn).to(d), "y": torch.from_numpy(y).to(d)}

    def __iter__(self):
        n = len(self.s)
        order = np.random.permutation(n) if self.shuffle else np.arange(n)
        for i in range(0, n, self.bs):
            yield self._tensors(order[i:i + self.bs])


def train_origins(b: Bundle, stride):
    o = np.arange(b.L, b.n_train - b.H + 1, stride)
    return np.repeat(np.arange(b.S), len(o)), np.tile(o, b.S)


def block_origins(b: Bundle, start):
    """Two non-overlapping H-blocks starting at ``start`` (val: n_train, test: n_val_end)."""
    return np.repeat(np.arange(b.S), 2), np.tile([start, start + b.H], b.S)


def future_origins(b: Bundle):
    return np.arange(b.S), np.full(b.S, b.T)


# ----------------------------------------------------------------------------
# Tabular features for XGBoost (lags / rolling stats shifted by >= H -> leak free)
# ----------------------------------------------------------------------------
TAB_FEATURES = ["store_c", "item_c", "dept_c", "cat_c", "scale", "dow", "dom", "month", "woy",
                "promo", "holiday", "price_rel"]


def build_tabular(b: Bundle) -> pd.DataFrame:
    S, T, H, sea = b.S, b.T, b.H, b.season
    Tx = T + H
    sales_ext = np.concatenate([b.sales, np.full((S, H), np.nan, np.float32)], 1)
    sdf = pd.DataFrame(sales_ext.T)                 # [Tx,S]
    sh = sdf.shift(H)
    feats = {}
    for k in [H, H + sea, H + 2 * sea, H + 4 * sea]:
        feats[f"lag_{k}"] = sdf.shift(k).values.T
    for w in (7, 28):
        feats[f"rmean_{w}"] = sh.rolling(w, min_periods=1).mean().values.T
    feats["rstd_28"] = sh.rolling(28, min_periods=2).std().values.T
    all_dates = b.dates.append(b.future_dates)
    flat = lambda a: np.asarray(a).reshape(-1)       # noqa: E731
    cal = {"dow": all_dates.dayofweek.values, "dom": all_dates.day.values, "month": all_dates.month.values,
           "woy": np.asarray(all_dates.isocalendar().week).astype(int)}
    d = {"series_idx": np.repeat(np.arange(S), Tx), "t": np.tile(np.arange(Tx), S), "y": flat(sales_ext)}
    for i, n in enumerate(["store_c", "item_c", "dept_c", "cat_c"]):
        d[n] = np.repeat(b.static_cat[:, i], Tx)
    d["scale"] = np.repeat(b.scale, Tx)
    for k, v in cal.items():
        d[k] = np.tile(v, S)
    d["promo"], d["holiday"], d["price_rel"] = flat(b.known[:, :, 6]), flat(b.known[:, :, 7]), flat(b.known[:, :, 8])
    for k, v in feats.items():
        d[k] = flat(v)
    return pd.DataFrame(d)


def tab_feature_names(b: Bundle):
    H, sea = b.H, b.season
    return (TAB_FEATURES + [f"lag_{k}" for k in [H, H + sea, H + 2 * sea, H + 4 * sea]] +
            ["rmean_7", "rmean_28", "rstd_28"])
