"""
paired_bootstrap_ci.py — Paired bootstrap CIs for the
federated-vs-centralized model gap, using the existing project artifacts.

Run from the project root:
    python src/federated/paired_bootstrap_ci.py

What it does
------------
1. Loads the canonical federated global test set via
   `nba_federated.task.load_global_test_set()` so the same test rows are
   scored by every model.
2. Loads the centralized baseline booster from
   `data/xgb_shot_model_tuned.json` (or retrains it on the federated train
   pool if `--retrain-central` is passed — recommended for a clean comparison,
   see notes at bottom of file).
3. Loads each per-seed federated best booster from
   `results/federated/xgb_federated_<agg>_<partition>_seed<S>_model_best.json`
   for the chosen aggregation × partition (default: bagging × team, the
   thesis primary configuration).
4. Reconciles feature sets between central and federated models automatically
   (the central model expects a subset of columns).
5. Computes paired bootstrap CIs (n_boot = 1000, seed = 42, percentile method)
   on:
      (a) Brier / log-loss / AUC point estimates for central and federated
      (b) Gap = federated − central, paired sample by sample
      (c) Across-seed mean federated, also against central
6. Writes a thesis-ready CSV (`results/federated/paired_bootstrap_ci.csv`)
   and prints a copy/paste table for Section 4.6.

Usage variants
--------------
    python src/federated/paired_bootstrap_ci.py                       # bagging × team
    python src/federated/paired_bootstrap_ci.py --agg cyclic --part iid
    python src/federated/paired_bootstrap_ci.py --retrain-central     # clean baseline
    python src/federated/paired_bootstrap_ci.py --n-boot 2000

Notes
-----
- The script auto-discovers all per-seed model JSONs that match the requested
  agg/partition. If a seed's `_best.json` is missing, that seed is skipped
  with a warning.
- Paired bootstrap: the same row indices are resampled for both models on
  each draw, so the gap CI is much tighter than independent bootstrapping.
- AUC bootstrap occasionally produces NaN if a resample contains only one
  class — those draws are dropped from the CI computation.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

# ──────────────────────────────────────────────────────────────────────
# Project paths — derive everything from this file's location so the
# script runs from the repo root or from the federated/ folder.
# ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent          # src/federated/
PROJECT_ROOT = SCRIPT_DIR.parents[1]                    # repo root
RESULTS_FED  = PROJECT_ROOT / "results" / "federated"
DATA_DIR     = PROJECT_ROOT / "data"

# Make the federated package importable when running from anywhere.
sys.path.insert(0, str(SCRIPT_DIR))
from nba_federated import task as fed_task  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────
EPS = 1e-15

def _brier(y, p):     return brier_score_loss(y, p)
def _logloss(y, p):   return log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])
def _auc(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, p)

METRICS = {"brier": _brier, "logloss": _logloss, "auc": _auc}
DIRECTION = {"brier": -1, "logloss": -1, "auc": +1}  # sign of "better"


# ──────────────────────────────────────────────────────────────────────
# Model loaders — handle both saved booster JSON and live retrain
# ──────────────────────────────────────────────────────────────────────
def _booster_from_json(path: Path) -> xgb.Booster:
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst


def _is_corrupt_json(path: Path) -> tuple[bool, str]:
    """
    Quick file-level sanity: empty, truncated, or non-JSON.
    Returns (is_corrupt, reason).
    """
    if not path.exists():
        return True, "file missing"
    size = path.stat().st_size
    if size == 0:
        return True, "empty file"
    if size < 200:
        return True, f"suspiciously small ({size} bytes — likely truncated)"
    try:
        head = path.read_bytes()[:32].lstrip()
        if not head.startswith(b"{"):
            return True, "does not start with '{' — not JSON"
    except OSError as e:
        return True, f"read failed: {e}"
    return False, ""


def _validate_predictions(p: np.ndarray, n_expected: int) -> tuple[bool, str]:
    """
    Reject predictions that are obviously broken: wrong length, NaN/inf,
    out-of-range, or degenerate (all the same constant)."""
    if p.shape != (n_expected,):
        return False, f"shape mismatch {p.shape} vs ({n_expected},)"
    if not np.isfinite(p).all():
        return False, "contains NaN/inf"
    if (p < 0).any() or (p > 1).any():
        return False, f"out-of-range probs [{p.min():.3f}, {p.max():.3f}]"
    if p.std() < 1e-6:
        return False, f"degenerate predictions (std={p.std():.2e})"
    return True, ""


def _load_with_fallback(path: Path, X_test_df: pd.DataFrame, n_test: int):
    """
    Try to load `path`. If it's corrupt or its predictions are invalid,
    fall back to the same filename without the `_best` suffix (the last-round
    booster, which is always saved server-side). Returns (booster_or_None,
    predictions_or_None, status_str).
    """
    # 1. Probe _best.json
    corrupt, reason = _is_corrupt_json(path)
    if not corrupt:
        try:
            bst = _booster_from_json(path)
            p = _predict_aligned(bst, X_test_df)
            ok, why = _validate_predictions(p, n_test)
            if ok:
                return bst, p, f"OK ({path.name})"
            reason = f"invalid predictions: {why}"
        except Exception as e:
            reason = f"load failed: {type(e).__name__}: {e}"

    # 2. Fallback to last-round model (drop the `_best` suffix)
    fallback = path.with_name(path.name.replace("_model_best.json", "_model.json"))
    if fallback.exists() and fallback != path:
        fb_corrupt, fb_reason = _is_corrupt_json(fallback)
        if not fb_corrupt:
            try:
                bst = _booster_from_json(fallback)
                p = _predict_aligned(bst, X_test_df)
                ok, why = _validate_predictions(p, n_test)
                if ok:
                    return bst, p, (
                        f"FALLBACK to last-round {fallback.name} "
                        f"(reason _best.json unusable: {reason})"
                    )
                return None, None, (
                    f"SKIPPED — _best.json {reason}; fallback {fallback.name} "
                    f"also bad: {why}"
                )
            except Exception as e:
                return None, None, (
                    f"SKIPPED — _best.json {reason}; fallback {fallback.name} "
                    f"load failed: {type(e).__name__}: {e}"
                )
        return None, None, (
            f"SKIPPED — _best.json {reason}; fallback {fallback.name} "
            f"also corrupt: {fb_reason}"
        )

    return None, None, f"SKIPPED — _best.json {reason}; no fallback file"

def _predict_aligned(bst: xgb.Booster, X_test_df: pd.DataFrame) -> np.ndarray:
    """
    Predict using a booster on a DataFrame whose columns may be a superset
    of the booster's expected feature names. Aligns columns by name.
    """
    expected = bst.feature_names
    if expected is None:
        # Booster without saved feature names — just trust column order.
        dmat = xgb.DMatrix(X_test_df.values)
        return bst.predict(dmat)
    missing = [c for c in expected if c not in X_test_df.columns]
    if missing:
        raise ValueError(
            f"Test set is missing features required by the booster: {missing}"
        )
    X_aligned = X_test_df[expected]
    dmat = xgb.DMatrix(X_aligned.values, feature_names=expected)
    return bst.predict(dmat)

def _retrain_central_on_fed_pool() -> tuple[xgb.Booster, list[str]]:
    """
    Train a centralized XGBoost (matching train_xgboost.py hyperparameters)
    on the federated framework's TRAIN pool, so the central booster has
    never seen any of the federated test games. This is the cleanest baseline
    for the federated-vs-central comparison.
    """
    df = fed_task.load_full_dataset()
    train_idx, _ = fed_task._get_global_split()
    feature_cols = fed_task._get_feature_cols(df)
    target_col = fed_task.TARGET_COL

    X_train = df.iloc[train_idx][feature_cols]
    y_train = df.iloc[train_idx][target_col]

    # Carve a validation slice for early stopping, group-aware on game_id.
    from sklearn.model_selection import GroupShuffleSplit
    if fed_task.GROUP_COL in df.columns:
        groups = df.iloc[train_idx][fed_task.GROUP_COL]
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
        tr, va = next(splitter.split(X_train, y_train, groups=groups))
        X_tr, X_va = X_train.iloc[tr], X_train.iloc[va]
        y_tr, y_va = y_train.iloc[tr], y_train.iloc[va]
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )

    print(f"[retrain-central] Train: {len(X_tr)} | Val: {len(X_va)}")

    clf = xgb.XGBClassifier(
        tree_method="hist",
        n_estimators=3000,
        learning_rate=0.02,
        max_depth=5,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
    bst = clf.get_booster()
    bst.feature_names = list(feature_cols)
    return bst, feature_cols


# ──────────────────────────────────────────────────────────────────────
# Bootstrap core
# ──────────────────────────────────────────────────────────────────────
def paired_bootstrap(
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """
    Paired percentile bootstrap. Returns point + (lo, hi) for each model
    and for the gap (B - A), per metric in METRICS.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    point = {m: {"A": fn(y, p_a), "B": fn(y, p_b)} for m, fn in METRICS.items()}
    for m in METRICS:
        point[m]["gap"] = point[m]["B"] - point[m]["A"]

    boots = {m: {"A": np.empty(n_boot), "B": np.empty(n_boot), "gap": np.empty(n_boot)}
             for m in METRICS}

    idx_all = np.arange(n)
    for i in range(n_boot):
        idx = rng.choice(idx_all, size=n, replace=True)
        y_b, pa_b, pb_b = y[idx], p_a[idx], p_b[idx]
        for m, fn in METRICS.items():
            a, b = fn(y_b, pa_b), fn(y_b, pb_b)
            boots[m]["A"][i] = a
            boots[m]["B"][i] = b
            boots[m]["gap"][i] = b - a

    lo_pct, hi_pct = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    out = {}
    for m in METRICS:
        a_arr = boots[m]["A"][~np.isnan(boots[m]["A"])]
        b_arr = boots[m]["B"][~np.isnan(boots[m]["B"])]
        g_arr = boots[m]["gap"][~np.isnan(boots[m]["gap"])]
        out[m] = {
            "central":    (point[m]["A"], *np.percentile(a_arr, [lo_pct, hi_pct])),
            "federated":  (point[m]["B"], *np.percentile(b_arr, [lo_pct, hi_pct])),
            "gap":        (point[m]["gap"], *np.percentile(g_arr, [lo_pct, hi_pct])),
            "n_eff":      len(g_arr),
        }
    return out


# ──────────────────────────────────────────────────────────────────────
# Seed discovery
# ──────────────────────────────────────────────────────────────────────
SEED_RE = re.compile(
    r"^xgb_federated_(?P<agg>bagging|cyclic)_(?P<part>iid|team)_seed(?P<seed>-?\d+)_model_best\.json$"
)

def discover_seed_models(agg: str, part: str) -> list[tuple[int, Path]]:
    """
    Find federated boosters for the chosen config.

    Returns a list of (seed_label, path) tuples. The non-seeded original run
    (xgb_federated_<agg>_<part>_model_best.json) is included as 'orig' since
    that is the booster the thesis Table 4.3 numbers come from.
    """
    out: list[tuple[int | str, Path]] = []

    # Original non-seeded run — what the thesis headline reports.
    orig = RESULTS_FED / f"xgb_federated_{agg}_{part}_model_best.json"
    if orig.exists():
        out.append(("orig", orig))

    for p in sorted(RESULTS_FED.glob(f"xgb_federated_{agg}_{part}_seed*_model_best.json")):
        m = SEED_RE.match(p.name)
        if m:
            out.append((int(m["seed"]), p))
    return out


def sanity_check_against_csv(seed_label, model_brier, model_auc):
    """
    Cross-check the loaded model's test metrics against the per-seed metrics CSV.
    Warn loudly if the loaded _best.json is not actually the best-AUC round
    (this means your server_app.py best-tracker fired too early).
    """
    if seed_label == "orig":
        csv = RESULTS_FED / "federated_bagging_team_metrics.csv"
    else:
        csv = RESULTS_FED / f"federated_bagging_team_seed{seed_label}_metrics.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    best_row = df.loc[df["auc"].idxmax()]
    matched = df.iloc[(df["brier"] - model_brier).abs().idxmin()]
    print(f"    [sanity] seed={seed_label}  loaded matches CSV round={int(matched['round'])} "
          f"(n_trees={int(matched['n_trees'])}, brier={matched['brier']:.4f}, "
          f"auc={matched['auc']:.4f})")
    if int(matched["round"]) != int(best_row["round"]):
        print(f"    [WARNING] _best.json is round {int(matched['round'])} but the actual "
              f"best-AUC round in the CSV is {int(best_row['round'])} "
              f"(n_trees={int(best_row['n_trees'])}, auc={best_row['auc']:.4f}, "
              f"brier={best_row['brier']:.4f}). Your server-side best-tracker is "
              f"saving the wrong checkpoint.")


# ──────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────
def fmt(x, d=4): return f"{x:.{d}f}"

def print_table(label: str, res: dict):
    print(f"\n=== {label} ===")
    print(f"{'metric':<8}  {'central [95% CI]':<28}  {'federated [95% CI]':<28}  {'gap (fed - central) [95% CI]':<32}  {'sig 95%':<8}")
    print("-" * 110)
    for m, r in res.items():
        c_pt, c_lo, c_hi = r["central"]
        f_pt, f_lo, f_hi = r["federated"]
        g_pt, g_lo, g_hi = r["gap"]
        sig = "yes" if (g_lo > 0 or g_hi < 0) else "no"
        print(f"{m:<8}  {fmt(c_pt)} [{fmt(c_lo)}, {fmt(c_hi)}]"
              f"  {fmt(f_pt)} [{fmt(f_lo)}, {fmt(f_hi)}]"
              f"  {fmt(g_pt)} [{fmt(g_lo)}, {fmt(g_hi)}]"
              f"  {sig:<8}")
    # Relative Brier degradation, matching the abstract / Section 4.6.1 wording.
    br = res["brier"]
    rel_pt = 100 * br["gap"][0] / br["central"][0]
    rel_lo = 100 * br["gap"][1] / br["central"][0]
    rel_hi = 100 * br["gap"][2] / br["central"][0]
    print(f"\nRelative Brier degradation: {rel_pt:+.2f}%  "
          f"95% CI [{rel_lo:+.2f}%, {rel_hi:+.2f}%]")


def results_to_long_df(label: str, res: dict) -> pd.DataFrame:
    rows = []
    for m, r in res.items():
        rows.append({
            "comparison": label, "metric": m,
            "central":   r["central"][0],   "central_lo":   r["central"][1],   "central_hi":   r["central"][2],
            "federated": r["federated"][0], "federated_lo": r["federated"][1], "federated_hi": r["federated"][2],
            "gap":       r["gap"][0],       "gap_lo":       r["gap"][1],       "gap_hi":       r["gap"][2],
            "sig_95":    bool(r["gap"][1] > 0 or r["gap"][2] < 0),
            "n_boot_effective": r["n_eff"],
        })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agg", choices=["bagging", "cyclic"], default="bagging")
    ap.add_argument("--part", choices=["team", "iid"], default="team")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--retrain-central", action="store_true",
                    help="Retrain centralized XGBoost on the federated train pool "
                         "for a clean baseline (recommended).")
    ap.add_argument("--central-json", type=str,
                    default=str(DATA_DIR / "xgb_shot_model_tuned.json"),
                    help="Path to saved centralized booster JSON (used when "
                         "--retrain-central is not set).")
    args = ap.parse_args()

    # 1. Canonical test set — same rows for every model
    print(f"Loading canonical federated global test set...")
    test_dmatrix, y_test = fed_task.load_global_test_set()
    df_full = fed_task.load_full_dataset()
    feature_cols_fed = fed_task._get_feature_cols(df_full)
    _, test_idx = fed_task._get_global_split()
    X_test = df_full.iloc[test_idx][feature_cols_fed].reset_index(drop=True)
    print(f"  N_test = {len(y_test)}   make rate = {y_test.mean():.4f}")

    # 2. Central baseline
    if args.retrain_central:
        print("Retraining centralized baseline on federated train pool...")
        central_bst, _ = _retrain_central_on_fed_pool()
        central_label = "central_retrained_on_fed_pool"
    else:
        print(f"Loading centralized baseline from {args.central_json}")
        central_path = Path(args.central_json)
        if not central_path.exists():
            sys.exit(f"ERROR: central booster not found at {central_path}. "
                     f"Pass --retrain-central or fix --central-json.")
        central_bst = _booster_from_json(central_path)
        central_label = central_path.name

    p_central = _predict_aligned(central_bst, X_test)
    print(f"  Central Brier  = {_brier(y_test, p_central):.4f}")
    print(f"  Central AUC    = {_auc(y_test, p_central):.4f}")

    # 3. Per-seed federated boosters
    seed_models = discover_seed_models(args.agg, args.part)
    if not seed_models:
        sys.exit(f"ERROR: no per-seed federated boosters found under "
                 f"{RESULTS_FED} for {args.agg}_{args.part}.")
    print(f"Found {len(seed_models)} federated seed(s) for {args.agg}_{args.part}: "
          f"{[s for s, _ in seed_models]}")

    fed_preds = []
    per_seed_summary = []
    long_rows = []
    skipped = []
    fallback_seeds = []

    for seed, path in seed_models:
        bst, p_fed, status = _load_with_fallback(path, X_test, len(y_test))
        print(f"  seed={seed:>4}: {status}")
        if bst is None:
            skipped.append((seed, status))
            continue
        if status.startswith("FALLBACK"):
            fallback_seeds.append(seed)

        fed_preds.append(p_fed)
        b = _brier(y_test, p_fed); a = _auc(y_test, p_fed); ll = _logloss(y_test, p_fed)
        per_seed_summary.append({
            "seed": seed, "brier": b, "logloss": ll, "auc": a,
            "source": "last_round" if seed in fallback_seeds else "best_round",
        })
        sanity_check_against_csv(seed, b, a)

    if skipped:
        print("\n  ----- skipped seeds (corrupt or missing) -----")
        for s, why in skipped:
            print(f"    seed={s}: {why}")

    if not fed_preds:
        sys.exit("ERROR: no usable federated boosters after corruption handling. "
                 "Re-run the federated experiments before bootstrapping.")

        # Per-seed paired bootstrap vs central
        res_seed = paired_bootstrap(
            y_test, p_central, p_fed,
            n_boot=args.n_boot, seed=args.seed,
        )
        long_rows.append(results_to_long_df(
            f"{args.agg}_{args.part}_seed{seed}_vs_{central_label}", res_seed
        ))

    print("\nPer-seed federated point estimates (on canonical test set):")
    print(pd.DataFrame(per_seed_summary).to_string(index=False))

    # 4. Across-seed mean prediction — paired bootstrap vs central
    P = np.stack(fed_preds, axis=0)  # (n_seeds, n_test)
    p_fed_mean = P.mean(axis=0)
    print("\n\n=== Across-seed paired bootstrap "
          f"({args.agg} × {args.part}, n_seeds = {len(seed_models)}) ===")
    res_mean = paired_bootstrap(
        y_test, p_central, p_fed_mean,
        n_boot=args.n_boot, seed=args.seed,
    )
    print_table(f"{args.agg} × {args.part}  —  fed_mean(over_seeds) vs {central_label}",
                res_mean)
    long_rows.append(results_to_long_df(
        f"{args.agg}_{args.part}_seed_mean_vs_{central_label}", res_mean
    ))

    # 5. Also report representative single seed (the one closest to seed mean)
    closest = int(np.argmin([
        np.mean((p - p_fed_mean) ** 2) for p in fed_preds
    ]))
    print(f"\nRepresentative single seed (closest to seed mean): "
          f"seed = {seed_models[closest][0]}")
    res_repr = paired_bootstrap(
        y_test, p_central, fed_preds[closest],
        n_boot=args.n_boot, seed=args.seed,
    )
    print_table(
        f"{args.agg} × {args.part}  —  seed={seed_models[closest][0]} vs {central_label}",
        res_repr,
    )

    # 6. Save consolidated CSV
    out_csv = RESULTS_FED / f"paired_bootstrap_ci_{args.agg}_{args.part}.csv"
    final_df = pd.concat(long_rows, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv.relative_to(PROJECT_ROOT)}")

    # 7. LaTeX fragment for direct inclusion in Section 4.6.1
    tex_path = RESULTS_FED / f"paired_bootstrap_ci_{args.agg}_{args.part}.tex"
    g = res_mean["brier"]["gap"]; c = res_mean["brier"]["central"][0]
    rel_pt = 100 * g[0] / c; rel_lo = 100 * g[1] / c; rel_hi = 100 * g[2] / c
    a_b = res_mean["brier"]
    a_l = res_mean["logloss"]
    a_a = res_mean["auc"]
    tex = (
        f"% Auto-generated by src/federated/paired_bootstrap_ci.py — "
        f"agg={args.agg}, partition={args.part}, n_boot={args.n_boot}\n"
        f"\\textbf{{Brier}}: centralized "
        f"${a_b['central'][0]:.4f}$ $[{a_b['central'][1]:.4f},\\,{a_b['central'][2]:.4f}]$, "
        f"federated $({args.agg}, {args.part})$ "
        f"${a_b['federated'][0]:.4f}$ $[{a_b['federated'][1]:.4f},\\,{a_b['federated'][2]:.4f}]$, "
        f"gap $\\Delta = {a_b['gap'][0]:+.4f}$ "
        f"$[{a_b['gap'][1]:+.4f},\\,{a_b['gap'][2]:+.4f}]$ "
        f"({rel_pt:+.2f}\\,\\% relative, "
        f"95\\,\\% CI $[{rel_lo:+.2f}\\,\\%,\\,{rel_hi:+.2f}\\,\\%]$).\n"
        f"\\textbf{{Log-loss}} gap $\\Delta = {a_l['gap'][0]:+.4f}$ "
        f"$[{a_l['gap'][1]:+.4f},\\,{a_l['gap'][2]:+.4f}]$. "
        f"\\textbf{{ROC-AUC}} gap $\\Delta = {a_a['gap'][0]:+.4f}$ "
        f"$[{a_a['gap'][1]:+.4f},\\,{a_a['gap'][2]:+.4f}]$.\n"
    )
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {tex_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
