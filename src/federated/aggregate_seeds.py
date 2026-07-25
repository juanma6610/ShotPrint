"""
aggregate_seeds.py — Reduce multi-seed federated runs into a mean±std table.

Reads every `federated_<agg>_<partition>_seed<S>_metrics.csv` produced by
`run_seeds.py`, picks each run's BEST round (max AUC), and aggregates
across seeds.

Output (printed and also saved as CSV):
  config              n_seeds   best_round_mean   AUC mean ± std       Brier mean ± std
  bagging_iid         5         14.4              0.6388 +/- 0.0021    0.2273 +/- 0.0006
  bagging_team        5         15.2              0.6395 +/- 0.0019    0.2270 +/- 0.0005
  cyclic_iid          5         512.6             ...
  cyclic_team         5         460.4             ...

Usage:
  python aggregate_seeds.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
RESULTS_DIR  = PROJECT_ROOT / "results" / "federated"

PATTERN = re.compile(
    r"^federated_(?P<agg>bagging|cyclic)_(?P<partition>iid|team)_seed(?P<seed>-?\d+)_metrics\.csv$"
)


def discover() -> pd.DataFrame:
    """Find all per-seed metrics CSVs and read each one's best round."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("federated_*_seed*_metrics.csv")):
        m = PATTERN.match(path.name)
        if not m:
            continue
        df = pd.read_csv(path)
        if df.empty:
            print(f"  WARNING: empty CSV {path.name}")
            continue
        best_row = df.loc[df["auc"].idxmax()]
        rows.append({
            "config":      f"{m['agg']}_{m['partition']}",
            "agg":         m["agg"],
            "partition":   m["partition"],
            "seed":        int(m["seed"]),
            "best_round":  int(best_row["round"]),
            "best_trees":  int(best_row["n_trees"]),
            "auc":         float(best_row["auc"]),
            "accuracy":    float(best_row["accuracy"]),
            "brier":       float(best_row["brier"]),
            "logloss":     float(best_row["logloss"]),
            "source":      path.name,
        })
    return pd.DataFrame(rows)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and std (population, ddof=1) across seeds, per config."""
    if per_seed.empty:
        return per_seed
    g = per_seed.groupby("config")
    out = pd.DataFrame({
        "n_seeds":          g.size(),
        "best_round_mean":  g["best_round"].mean().round(1),
        "best_trees_mean":  g["best_trees"].mean().round(1),
        "auc_mean":         g["auc"].mean().round(4),
        "auc_std":          g["auc"].std(ddof=1).round(4),
        "brier_mean":       g["brier"].mean().round(4),
        "brier_std":        g["brier"].std(ddof=1).round(4),
        "logloss_mean":     g["logloss"].mean().round(4),
        "logloss_std":      g["logloss"].std(ddof=1).round(4),
    }).reset_index()
    return out


def format_table(summary: pd.DataFrame) -> str:
    lines = []
    header = f"{'config':<16} {'n':>3}  {'best_rd':>8}  {'AUC':>20}  {'Brier':>20}  {'LogLoss':>20}"
    lines.append(header)
    lines.append("-" * len(header))
    for _, r in summary.iterrows():
        auc   = f"{r['auc_mean']:.4f} +/- {r['auc_std']:.4f}"     if not np.isnan(r['auc_std'])   else f"{r['auc_mean']:.4f}    (n=1)"
        brier = f"{r['brier_mean']:.4f} +/- {r['brier_std']:.4f}" if not np.isnan(r['brier_std']) else f"{r['brier_mean']:.4f}    (n=1)"
        ll    = f"{r['logloss_mean']:.4f} +/- {r['logloss_std']:.4f}" if not np.isnan(r['logloss_std']) else f"{r['logloss_mean']:.4f}    (n=1)"
        lines.append(f"{r['config']:<16} {r['n_seeds']:>3}  {r['best_round_mean']:>8.1f}  {auc:>20}  {brier:>20}  {ll:>20}")
    return "\n".join(lines)


def main():
    print(f"Results directory: {RESULTS_DIR}")
    per_seed = discover()
    if per_seed.empty:
        print("No per-seed metrics CSVs found. Did you run run_seeds.py yet?")
        return 1

    print(f"Found {len(per_seed)} per-seed runs across "
          f"{per_seed['config'].nunique()} configs")
    print()

    per_seed_out = RESULTS_DIR / "_aggregated_per_seed.csv"
    per_seed.to_csv(per_seed_out, index=False)

    summary = summarise(per_seed)
    summary_out = RESULTS_DIR / "_aggregated_summary.csv"
    summary.to_csv(summary_out, index=False)

    print(format_table(summary))
    print()
    print(f"Per-seed table : {per_seed_out}")
    print(f"Summary table  : {summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
