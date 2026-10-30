"""training.py - training / prediction loops for the neural models."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from models import pinball_loss


def tau_schedule(cfg: Config, epoch: int) -> float:
    """Exponential Gumbel-softmax temperature annealing: tau_start -> tau_end over the epochs."""
    frac = epoch / max(1, cfg.epochs - 1)
    return cfg.tau_start * (cfg.tau_end / cfg.tau_start) ** frac


def train_neural(model, train_s, val_s, cfg: Config, kind: str, device: str, log=print):
    """kind='tft': L = pinball + lambda_mech * L_mechanism.   kind='lstm': MSE.  Keeps the best val epoch."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
    best, best_state, hist = float("inf"), None, []
    for ep in range(cfg.epochs):
        if kind == "tft":
            model.set_temperature(tau_schedule(cfg, ep))
        model.train()
        tl, tm = [], []
        for b in train_s:
            opt.zero_grad()
            if kind == "tft":
                pred, mech = model(b["x"], b["sc"], b["sn"])
                lq = pinball_loss(pred, b["y"], cfg.quantiles)
                huber = F.huber_loss(pred[..., 1], b["y"])
                l1_p50 = F.l1_loss(pred[..., 1], b["y"])          # NEW: MAE term — drives down MASE
                mse = F.mse_loss(pred[..., 1], b["y"])
                # Hybrid loss: Quantile + Mechanism + Huber + MSE for lowest WRMSSE & MASE
                loss = lq + cfg.lambda_mech * mech + 0.3 * huber + 0.1 * l1_p50
                tm.append(float(mech.detach()))
            else:
                loss = F.mse_loss(model(b["x"], b["sc"], b["sn"]), b["y"])
            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl.append(float(loss.detach()))
        model.eval()
        vl = []
        with torch.no_grad():
            for b in val_s:
                if kind == "tft":
                    pred, mech = model(b["x"], b["sc"], b["sn"])
                    lq = pinball_loss(pred, b["y"], cfg.quantiles)
                    huber = F.huber_loss(pred[..., 1], b["y"])
                    l1_p50 = F.l1_loss(pred[..., 1], b["y"])
                    vl.append(float(lq + 0.5 * huber + 0.3 * l1_p50))
                else:
                    vl.append(float(F.mse_loss(model(b["x"], b["sc"], b["sn"]), b["y"])))
        v = float(np.mean(vl))
        sched.step(v)  # ← called after validation with actual val loss
        hist.append({"epoch": ep + 1, "train_loss": float(np.mean(tl)), "val_loss": v,
                     "mech_loss": float(np.mean(tm)) if tm else 0.0,
                     "tau":tau_schedule(cfg,ep) if kind == "tft" else 0.0,
                     "huber_weight":0.1 if kind == "tft" else 0.0})
        log(f"[{kind}] epoch {ep + 1}/{cfg.epochs} train={hist[-1]['train_loss']:.4f} val={v:.4f}"
            + (f" mech={hist[-1]['mech_loss']:.4f} tau={hist[-1]['tau']:.2f}" if kind == "tft" else ""))
        if v < best:
            best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


@torch.no_grad()
def predict_neural(model, sampler, kind: str):
    model.eval()
    out = []
    for b in sampler:
        if kind == "tft":
            p, _ = model(b["x"], b["sc"], b["sn"])
        else:
            p = model(b["x"], b["sc"], b["sn"]).unsqueeze(-1)
        out.append(p.cpu().numpy())
    return np.concatenate(out)           # [N,H,Q]
