"""train.py - CLI: run the full pipeline and save artifacts for the dashboard.

    python train.py                      # dummy data, quick demo
    python train.py --source m5 --data_dir ./m5 --max_series 300 --epochs 15
"""
import argparse
from pathlib import Path

import torch

from config import Config
from pipeline import run_pipeline, save_results


def main():
    ap = argparse.ArgumentParser()
    d = Config()
    ap.add_argument("--source", default=d.source, choices=["dummy", "m5", "walmart", "favorita"])
    ap.add_argument("--data_dir", default=d.data_dir)
    ap.add_argument("--max_series", type=int, default=d.max_series)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--horizon", type=int, default=d.horizon)
    ap.add_argument("--encoder_len", type=int, default=d.encoder_len)
    ap.add_argument("--freq", default=d.freq)
    ap.add_argument("--season", type=int, default=d.season)
    ap.add_argument("--lambda_mech", type=float, default=d.lambda_mech)
    ap.add_argument("--topk_ratio", type=float, default=d.topk_ratio)
    ap.add_argument("--out", default=d.artifacts_dir)
    a = ap.parse_args()
    cfg = Config(source=a.source, data_dir=a.data_dir, max_series=a.max_series, epochs=a.epochs, horizon=a.horizon,
                 encoder_len=a.encoder_len, freq=a.freq, season=a.season, lambda_mech=a.lambda_mech,
                 topk_ratio=a.topk_ratio, artifacts_dir=a.out)
    res = run_pipeline(cfg)
    out = Path(a.out)
    save_results(res, out / "results.pkl")
    torch.save(res["_weights"]["tft"], out / "tft_sparse.pt")
    res["metrics"].to_csv(out / "metrics.csv", index=False)
    res["inv"]["schedule"].to_csv(out / "optimization_schedule.csv", index=False)
    print("\n", res["metrics"].round(4).to_string(index=False))
    print(f"\nArtifacts saved in {out.resolve()}")


if __name__ == "__main__":
    main()
