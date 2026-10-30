"""
models.py - Contract-Sparse Transformer (primary) + Naive / XGBoost / LSTM baselines.

NOTE ON NAMING
--------------
The architecture in the diagrams is a *plain Transformer encoder* with a custom sparse-attention block.
It is NOT a Temporal Fusion Transformer (no variable-selection networks / GRNs / LSTM seq2seq), so it is
called ``ContractSparseTransformer`` here.

Data flow (matches the three diagrams)
--------------------------------------
Historical sales + known covariates -> Input Embedding -> Positional Encoding
SKU meta features -> Task Embedding Generator (SKUContextProvider) ---> context vector c
for each encoder layer:
    Layer Norm 1 -> ContractSparseAttention(h, c) -> Add&Norm 1 -> FFN -> Add&Norm 2
Output Projection (last H positions) -> monotone (P10 <= P50 <= P90) -> Quantile loss
Mechanism loss (from every attention block) is added:   L = L_quantile + lambda * L_mechanism
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Building blocks
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SKUContextProvider(nn.Module):
    """SKU Meta Features -> Task Embedding Generator. Embeds store / item / dept / category ids (id 0 = unknown)
    and numeric SKU statistics, then an MLP produces one context vector ``c`` per series."""

    def __init__(self, cat_cards, n_num, d_model, d_emb=8):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c + 1, d_emb, padding_idx=0) for c in cat_cards])
        self.mlp = nn.Sequential(nn.Linear(len(cat_cards) * d_emb + n_num, d_model), nn.GELU(),
                                 nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, static_cat, static_num):
        e = [emb(static_cat[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [static_num], -1))


# ============================================================================
# Core module: Contract-Theoretic Sparse Attention (Meta-Adaptive Query Selection)
# ============================================================================
class ContractSparseAttention(nn.Module):
    """
    Contract-theoretic query selection + sparse attention.

    Mechanism-design analogy
    ------------------------
    * Every *query* (time step) is an agent with a private **type** (how much it needs to attend).
      ``type_net`` (Type Estimation Net) outputs a distribution over ``M`` ordered types with values
      v_1 < ... < v_M.
    * The **Contract Designer** (``menu_net``, conditioned on the SKU context ``c``) outputs a *menu* of M
      contracts ``(a_m, p_m)``: allocation a_m in (0,1] (attention budget share, increasing in m via a cumulative
      softmax) and payment p_m >= 0 (increasing via cumulative softplus).
    * Type m has utility   U(m, m') = v_m * a_{m'} - p_{m'}   for choosing contract m'.
    * A query's expected allocation  a_q = sum_m P(type=m | q) * a_m  biases its selection score.
    * The **Importance Scoring Network** gives a data-driven score s_q.  Selection score:
          u_q = s_q + log(a_q)
    * **Top-K selector** (K = ceil(topk_ratio * T) = the attention budget) with Gumbel noise.
      Forward pass: HARD top-K mask.  Backward pass: gradient of a SOFT Gumbel-softmax relaxation
      (straight-through estimator):   mask = hard + soft - soft.detach(),
      soft = 1 - exp(-K * softmax((u + g) / tau)).   tau is annealed by the trainer (``set_temperature``).
    * **Sparse attention**: selected queries get full softmax attention over keys (+1 static key/value token
      produced by the *Static Key Value Projectors* from the SKU context). Non-selected ("lazy") queries
      receive the mean of V (Informer-style), so they still carry global context.

    Mechanism loss (``aux['mech']``)
    --------------------------------
        L_mech = L_IC + L_IR + w_sp * L_sparsity
        L_IC  = mean_m  relu( max_m' U(m,m') - U(m,m) )     incentive-compatibility violation
        L_IR  = mean_m  relu( -U(m,m) )                      individual-rationality violation
        L_sp  = mean entropy(selection distribution)/log(T)  pushes towards decisive (sparse) selection
    Total training loss:   L = L_quantile + lambda_mech * L_mech.

    Implementation note: for clarity the dense T x T score matrix is computed and then masked (query-wise);
    this reproduces the sparse semantics exactly but does not save FLOPs. A gather-based kernel would be needed
    for real speed-ups on very long sequences.
    """

    def __init__(self, d_model, n_heads, n_types=4, topk_ratio=0.6, dropout=0.1, force_last=0, sparsity_weight=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.d = n_heads, d_model // n_heads, d_model
        self.M, self.ratio, self.force_last, self.sp_w = n_types, topk_ratio, force_last, sparsity_weight
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(d_model, d_model) for _ in range(4))
        # Static Key / Value projectors (SKU context -> one extra key & value token)
        self.sk_proj, self.sv_proj = nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)
        # Type Estimation Net, Importance Scoring Network, Contract Menu generator
        self.type_net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, n_types))
        self.importance_net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.menu_net = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2 * n_types))
        self.register_buffer("type_values", torch.linspace(0.25, 1.0, n_types))
        self.drop = nn.Dropout(dropout)
        self.tau = 1.0

    # ------------------------------------------------------------------
    def contract_menu(self, ctx):
        raw = self.menu_net(ctx).view(-1, self.M, 2)
        alloc = torch.cumsum(torch.softmax(raw[..., 0], -1), -1)        # (0,1], increasing, last == 1
        pay = torch.cumsum(F.softplus(raw[..., 1]), -1)                 # >=0, increasing
        return alloc, pay

    def forward(self, h, ctx):
        B, T, D = h.shape
        q = self.q_proj(h).view(B, T, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(h).view(B, T, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(h).view(B, T, self.h, self.dk).transpose(1, 2)
        ks = self.sk_proj(ctx).view(B, 1, self.h, self.dk).transpose(1, 2)
        vs = self.sv_proj(ctx).view(B, 1, self.h, self.dk).transpose(1, 2)
        k, v = torch.cat([ks, k], 2), torch.cat([vs, v], 2)              # index 0 = static token

        # ---------------- Contract based query selection ----------------
        type_p = torch.softmax(self.type_net(h), -1)                      # [B,T,M]   Type Estimator
        alloc, pay = self.contract_menu(ctx)                              # [B,M]     Contract Designer
        q_alloc = (type_p * alloc[:, None, :]).sum(-1)                    # [B,T]
        imp = self.importance_net(h).squeeze(-1)                          # [B,T]     Importance scorer
        u = imp + torch.log(q_alloc + 1e-6)
        if self.force_last > 0:                                           # optional: always keep forecast positions
            bonus = torch.zeros_like(u)
            bonus[:, -self.force_last:] = 1e3
            u = u + bonus
        K = max(1, int(math.ceil(self.ratio * T)))
        if self.training:                                                 # Gumbel perturbation
            g = -torch.log(-torch.log(torch.rand_like(u).clamp(1e-9, 1 - 1e-9)))
            y = u + g
        else:
            y = u
        hard = torch.zeros_like(u).scatter(1, y.topk(K, dim=1).indices, 1.0)
        p = torch.softmax(y / self.tau, dim=1)
        soft = 1.0 - torch.exp(-K * p)
        mask = hard + soft - soft.detach()                                # straight-through estimator

        # ---------------- Sparse attention computation ------------------
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dk)  # [B,h,T,T+1]
        attn = self.drop(torch.softmax(scores, -1))                          # softmax normalisation
        full = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, D)        # weighted sum with V
        lazy = v[:, :, 1:, :].mean(2).reshape(B, 1, D)                       # mean-V for non-selected queries
        m = mask.unsqueeze(-1)
        out = self.o_proj(m * full + (1.0 - m) * lazy)                       # masked (query-wise) attention

        # ---------------- Mechanism loss ---------------------------------
        U = self.type_values[None, :, None] * alloc[:, None, :] - pay[:, None, :]   # [B,type,contract]
        own = U.diagonal(dim1=1, dim2=2)
        ic = F.relu(U.max(-1).values - own).mean()
        ir = F.relu(-own).mean()
        ent = -(p * torch.log(p + 1e-9)).sum(1) / math.log(T)
        mech = ic + ir + self.sp_w * ent.mean()
        return out, {"mech": mech, "ic": ic.detach(), "ir": ir.detach(), "entropy": ent.mean().detach()}


class SparseEncoderLayer(nn.Module):
    """Layer Norm 1 -> Contract-Sparse Attention -> Add&Norm 1 -> Feed-Forward -> Add&Norm 2."""

    def __init__(self, d_model, n_heads, d_ff, dropout, n_types, topk_ratio, force_last):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ContractSparseAttention(d_model, n_heads, n_types, topk_ratio, dropout, force_last)
        self.an1, self.an2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, ctx):
        a, aux = self.attn(self.ln1(x), ctx)
        x = self.an1(x + self.drop(a))
        x = self.an2(x + self.drop(self.ffn(x)))
        return x, aux


class ContractSparseTransformer(nn.Module):
    """Primary probabilistic model. Output ``preds [B,H,3]`` = (P10, P50, P90), non-crossing by construction
    (P10 = P50 - softplus(.), P90 = P50 + softplus(.)); also returns the mean mechanism loss."""

    def __init__(self, n_features, cat_cards, n_num, horizon, d_model=48, n_heads=4, n_layers=2, d_ff=96,
                 dropout=0.1, n_types=4, topk_ratio=0.6, force_last=0):
        super().__init__()
        self.H = horizon
        self.in_embed = nn.Linear(n_features, d_model)                    # Input Embedding Layer
        self.pos = PositionalEncoding(d_model)
        self.sku = SKUContextProvider(cat_cards, n_num, d_model)          # SKU Context Provider
        self.ctx_to_tok = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([SparseEncoderLayer(d_model, n_heads, d_ff, dropout, n_types, topk_ratio, force_last)
                                     for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, 3)                             # Output Projection Layer

    def set_temperature(self, tau):
        for layer in self.layers:
            layer.attn.tau = float(tau)

    def forward(self, x, static_cat, static_num):
        ctx = self.sku(static_cat, static_num)
        h = self.pos(self.in_embed(x) + self.ctx_to_tok(ctx).unsqueeze(1))
        mech = x.new_zeros(())
        for layer in self.layers:
            h, aux = layer(h, ctx)
            mech = mech + aux["mech"]
        o = self.out_proj(h[:, -self.H:, :])
        p50 = o[..., 0]
        p10 = p50 - F.softplus(o[..., 1])                                  # monotone quantiles (no crossing)
        p90 = p50 + F.softplus(o[..., 2])
        return torch.stack([p10, p50, p90], -1), mech / len(self.layers)


def pinball_loss(pred, y, quantiles=(0.1, 0.5, 0.9)):
    """Quantile (pinball) loss, pred [B,H,Q], y [B,H]."""
    q = torch.tensor(quantiles, device=pred.device, dtype=pred.dtype)
    e = y.unsqueeze(-1) - pred
    return torch.maximum(q * e, (q - 1) * e).mean()


# ============================================================================
# Baselines
# ============================================================================
class LSTMBaseline(nn.Module):
    """Point-forecast LSTM using the same window layout (history + known future covariates)."""

    def __init__(self, n_features, cat_cards, n_num, horizon, hidden=64, d_emb=8):
        super().__init__()
        self.H = horizon
        self.embs = nn.ModuleList([nn.Embedding(c + 1, d_emb, padding_idx=0) for c in cat_cards])
        self.lstm = nn.LSTM(n_features + len(cat_cards) * d_emb + n_num, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, static_cat, static_num):
        s = torch.cat([e(static_cat[:, i]) for i, e in enumerate(self.embs)] + [static_num], -1)
        inp = torch.cat([x, s.unsqueeze(1).expand(-1, x.size(1), -1)], -1)
        out, _ = self.lstm(inp)
        return self.head(out[:, -self.H:, :]).squeeze(-1)                # [B,H]


class XGBBaseline:
    """Global gradient-boosted trees on leak-free tabular features (lags / rolling stats shifted >= H)."""

    def __init__(self, feature_names, **kw):
        import xgboost as xgb
        self.features = list(feature_names)
        params = dict(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                      tree_method="hist", early_stopping_rounds=30, n_jobs=4)
        params.update(kw)
        self.model = xgb.XGBRegressor(**params)

    def fit(self, tr, va):
        self.model.fit(tr[self.features], tr["y"], eval_set=[(va[self.features], va["y"])], verbose=False)
        return self

    def predict(self, df):
        return np.clip(self.model.predict(df[self.features]), 0, None)


def naive_forecast(sales, origin, horizon, season=7):
    """Seasonal-naive: repeat the last ``season`` observed steps (strongest simple benchmark for retail).
    Uses only data strictly before ``origin``.  Returns [S,H]."""
    h = np.arange(horizon)
    return sales[:, origin - season + (h % season)]
