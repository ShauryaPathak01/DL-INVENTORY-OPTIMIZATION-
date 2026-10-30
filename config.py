"""config.py - single source of truth for every hyper-parameter in the project."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    source: str = "m5"
    data_dir: str = "m5"
    freq: str = "D"
    max_series: int = 100
    dummy_stores: int = 3
    dummy_items: int = 10
    dummy_days: int = 730

    # --------------------------------------------------------------- windows
    horizon: int = 14              # H  : forecast length
    encoder_len: int = 28          # L  : history length fed to the encoder
    season: int = 7                # seasonal period (naive forecast, lags)
    train_stride: int = 2          # stride between training window origins
    seed: int = 42

    # ----------------------------------------------- Contract-Sparse Transformer
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 128
    dropout: float = 0.05
    n_types: int = 6              # number of query "types" / contracts on the menu
    topk_ratio: float = 0.5        # fraction of queries allowed to attend (the "budget")
    force_select_horizon: bool = True  # if True the H forecast positions are always selected
    tau_start: float = 2.0         # Gumbel-softmax temperature, annealed exponentially
    tau_end: float = 0.2
    lambda_mech: float = 0.02      # L = L_quantile + lambda_mech * L_mechanism
    quantiles: tuple = (0.1, 0.5, 0.9)

    # ------------------------------------------------------------- training
    epochs: int = 12
    batch_size: int = 512
    lr: float = 1e-3
    lstm_hidden: int = 64

    # ------------------------------------------------------------ inventory
    lead_time: int = 3             # days between ordering and receiving
    review_period: int = 1         # days between reviews
    service_level: float = 0.95    # target cycle service level
    rho: float = 0.2               # day-to-day forecast-error correlation (0 = indep., 1 = perfect)
    holding_rate_per_day: float = 0.002   # fraction of unit cost per unit per day
    cost_ratio: float = 0.7        # unit cost = cost_ratio * unit price
    goodwill_mult: float = 0.5     # stock-out penalty = margin * (1 + goodwill_mult)
    order_cost: float = 1.0        # fixed cost per purchase order line
    overstock_threshold: float = 0.5   # leftover > 50 % of window demand counts as overstock

    # ------------------------------------------------------------- analytics
    shap_samples: int = 40
    artifacts_dir: str = "artifacts"

    # derived split lengths (all in time steps) -----------------------------
    @property
    def val_days(self) -> int:
        return 2 * self.horizon     # two non-overlapping forecast blocks

    @property
    def test_days(self) -> int:
        return 2 * self.horizon

    def to_dict(self) -> dict:
        return asdict(self)
