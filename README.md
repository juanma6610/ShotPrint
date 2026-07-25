# Reading the Floor — A Portable Federated Shot Quality Model from Optical Tracking

Code and data for the Master's thesis of **Juan Manuel Oliver** (MSc Artificial Intelligence, Big Data Analytics, KU Leuven, 2025–26). The project builds a calibrated NBA shot-make-probability model from 2015–16 SportVU optical tracking + play-by-play data, trains it both centrally and under a cross-silo **federated learning** protocol (30 teams as clients), and derives downstream applications: **Points Over Expectation (POE)** for shooters, defenders, zones, and 5-man lineups.

## Pipeline at a glance

```
allgames.txt (636 game archives)
        │
        ▼
[1] clusters/ (R) ──── scrape.Rmd → shooters.Rmd / defense.Rmd
        │              GMM soft archetypes (4 offensive + 4 defensive)
        │              → gmm_soft_labels_15_16.csv / gmm_soft_labels_def_15_16.csv
        ▼
[2] src/process_batch.py ── downloads each game, runs shot_features.py
        │                   (release-frame recovery, geometry, kinematics,
        │                    defender pressure, spacing, tempo, archetypes)
        │                   → data/shot_features_full.csv → …_valid2.csv
        ▼
[3] src/train_xgboost.py / tune_xgboost.py ── centralized XGBoost baseline
        │                                     → data/xgb_shot_model(.tuned).json
        ▼
[4] src/compute_poe.py → compute_def_poe.py / correlate_poe.py
        │                out-of-fold POE, DEF-POE, matchup heatmap,
        │                correlations with Basketball-Reference metrics
        ▼
[5] src/federated/ (Flower) ── bagging/cyclic × team/IID × 5 seeds
        │                      → results/federated/*
        ▼
[6] src/thesis_results.py + figures_thesis/scripts/ ── thesis tables & figures
```

## Repository layout

Top level:

| Path | What it is |
|---|---|
| `allgames.txt` | 636 SportVU game archive names (2015–16 regular season), consumed by `process_batch.py`. |
| `requirements.txt` | Python dependencies (xgboost, flwr, scikit-learn, py7zr, …). |
| `README.md` | This file. |
| `src/` | Python pipeline (feature extraction → model → POE → federated). |
| `clusters/` | R project for player-archetype clustering. |
| `data/` | Datasets, saved models, and `data/legacy/` (superseded files). |
| `results/` | Metric tables, POE leaderboards, tuning history, and `results/federated/`. |
| `figures/` | All generated figures (result figures + thesis figure outputs, consolidated here). |
| `figures_thesis/` | Static thesis image assets + `scripts/` that regenerate figures into `figures/`. |
| `docs/` | Reference material: the ydata-profiling EDA report. |

### `clusters/` — player archetype clustering (R / RStudio)

Standalone R project (`renv` lockfile included) that produces the soft archetype features used by the Python pipeline.

| File | Purpose |
|---|---|
| `scrape.Rmd` | Scrapes NBA.com hidden stats API for player-tracking defense tables → `data/defensive_tracking.csv`. |
| `shooters.Rmd` | Offensive archetypes: PCA + K-means + **GMM (BIC-selected K=4)** on shot-creation profile (USG, 3PAr, %Ast, FTr, FG% by zone). Outputs `shooter_archetypes_15_16.csv`, `gmm_soft_labels_15_16.csv` (Primary_Creator, Spacer, Mid-Interior, Rim_Center). |
| `defense.Rmd` | Defensive archetypes, same method. Outputs `defender_archetypes_15_16.csv`, `gmm_soft_labels_def_15_16.csv` (Paint_Anchors, Perimeter Guards, Def_liability, Switch_Wing). |
| `cluster_report.Rmd/.html` | Write-up of the clustering results. |
| `data/` | Basketball-Reference exports: `ad.csv`/`pos.csv`/`shot.csv` (2015–16), `ad_14.csv`/`pos_14.csv`/`shot_14.csv` (2014–15, for skill priors), `defensive_tracking.csv`, `opp_shooting_by_zone.csv`. |
| `pca_loadings.csv`, `gmm_*_profiles_*.csv` | Cluster interpretation artifacts. |

### `src/` — main Python pipeline

| File | Purpose |
|---|---|
| `game.py` | `Game` class: downloads a game's SportVU 7z from the `sealneaward/nba-movement-data` mirror + its PBP CSV, parses moments into a DataFrame, aligns clocks, exposes helpers (frames, commentary, formation detection). |
| `kinematics.py` | Savitzky-Golay velocity/acceleration estimation per player; `time_to_reach` closeout-time model (max speed 22 ft/s, max accel 10 ft/s²). |
| `spatial.py` | Team convex-hull spacing, Voronoi space control, delta-distance/delta-time control maps. |
| `shot_features.py` | Core extractor. For each PBP shot event: recovers the release frame (ball-z apex ≥ 9 ft, walk-back to z ≤ 10 ft & ball–shooter dist ≤ 2.5 ft), then computes ~37 model features: geometry (dist, x, y, angle), defender pressure (closest/second defender distance-angle-time, tight-contest counts), kinematic decomposition (parallel/perpendicular velocity & acceleration for shooter and defender), release mechanics (height, speed, angle, x, y), tempo (shot clock, touch time, catch-and-shoot), spacing (hull ratio), and the 8 GMM archetype probabilities. Dunks/tips flagged via PBP regex. |
| `process_batch.py` | Multiprocessing batch driver over `allgames.txt` → `data/shot_features_full.csv` (rename/copy to `_valid2` after archetype columns are applied). |
| `train_xgboost.py` | Centralized baseline. Game-disjoint 65/15/20 split (`GroupShuffleSplit` on `game_id`), early stopping on validation log-loss, evaluation focused on calibration (Brier, log loss, reliability diagram). |
| `tune_xgboost.py` | Random search (30 candidates × GroupKFold(5)) scored by CV log-loss; refits best config → `data/xgb_shot_model_tuned.json`, `results/best_params.json`. |
| `compute_poe.py` | Out-of-fold (GroupKFold over games) P(make) for every shot → per-shot POE (`shot_value × (made − xMake)`, shot value from PBP "3PT" regex) → player leaderboard (≥100 shots). |
| `compute_def_poe.py` | Defender POE (points suppressed below expectation for the closest defender) + offensive-vs-defensive archetype matchup heatmap. |
| `correlate_poe.py` | Correlates POE with Basketball-Reference advanced metrics (TS%, PER, OBPM, …) incl. partial correlations controlling for USG%. Reads `results/poe_leaderboard.csv`; writes correlation tables to `results/`. |
| `thesis_results.py` | One-shot generator of Results-chapter tables and figures (headline metrics vs baselines, per-zone metrics, calibration, feature importance, POE leaderboard, case-study shot charts). Reads `data/shot_features_valid2.csv`. |
| `plot_per_zone_poe.py` | Per-zone POE decomposition figure (Results section). |
| `visualization.py` | Court drawing, frame rendering, game animation utilities (used for Figure 3.1-style renders). |
| `spa.ipynb` | Spacing analysis notebook; also builds the lineup POE tables (`data/lineup_poe*.csv`). |

### `src/federated/` — Flower federated XGBoost

| File | Purpose |
|---|---|
| `nba_federated/task.py` | Data loading/partitioning from `data/shot_features_valid2.csv`. Single cached game-disjoint global 85/15 train/test split (seed 42, shared by all clients & server); partitions: `team` (30 non-IID clients = franchises) or `iid` (stratified control). |
| `nba_federated/client_app.py` | Flower client: trains 1 tree/round on local data from the global booster. |
| `nba_federated/server_app.py` | Flower server: `FedXgbBagging` (primary) or `FedXgbCyclic` (ablation, with a custom central-eval adapter); per-round central evaluation, best-AUC checkpointing → `results/federated/`. |
| `pyproject.toml` | Flower app config + XGBoost hyperparameters (depth 4, eta 0.01, min_child_weight 10, λ 2.0). |
| `run_seeds.py` | Runs the 4 configs (bagging/cyclic × iid/team) × 5 seeds {42, 7, 123, 2024, 99}, renaming outputs per run. |
| `aggregate_seeds.py` | Reduces per-seed metrics to the mean±std table (thesis Table 4.3). |
| `paired_bootstrap_ci.py` | Paired bootstrap CIs (1000 resamples) for the federated-vs-centralized gap → `paired_bootstrap_ci_*.csv/.tex`. |
| `implementation_plan_federated.md`, `federated_learning_discussion.md` | Design notes. |

### `data/` — datasets and models

| File | Purpose |
|---|---|
| `tracking/` | Three sample raw SportVU game JSONs. |
| `shot_features_valid2.csv` | **Canonical training file** — 95,219 shots × 48 cols, archetypes as the named 4+4 GMM soft labels. Used by `train_xgboost.py`, `tune_xgboost.py`, `compute_poe.py`, and the federated pipeline. |
| `shot_features_valid.csv` | Earlier extraction pass: 97,826 shots / 630 games (the counts quoted in the thesis data section). |
| `xgb_shot_model_tuned.json` | Saved tuned centralized booster. |
| `lineup_poe.csv`, `lineup_poe_offense.csv`, `lineup_poe_defense.csv` | 5-man lineup POE tables (built in `spa.ipynb`). |
| `players.csv`, `pbp/` | Support lookups / sample PBP + `EVENTMSGTYPE` code reference (`pbpevents.txt`). |
| `legacy/` | Superseded artifacts kept for reference: `shot_features.csv`, `shot_features_before_dunks.csv`, untuned `xgb_shot_model.json`, and the older `poe_*` copies. Git-ignored. |

### `results/` — experiment outputs

Canonical POE outputs (`poe_per_shot.csv`, `poe_leaderboard.csv`), headline/per-zone metric tables, tuning history + best params, correlations with advanced metrics, and `federated/` with per-seed metrics CSVs, per-seed best boosters, aggregated summary, and paired-bootstrap CI tables (CSV + LaTeX).

### `figures/` and `figures_thesis/`

`figures/` is the single output directory for every generated figure (calibration, convergence, heatmaps, leaderboards, plus all thesis `build_*` outputs — PNG + PDF). `figures_thesis/` holds static image assets (broadcast stills, player photos) and `scripts/` — one `build_*.py` per thesis figure (pipeline architecture, release-frame recovery, kinematics decomposition, SHAP beeswarm, dunk anomaly, lineup leaderboards, POE time series, xFG scatter, etc.), each now writing into `figures/`.

## Reproducing the pipeline

```bash
pip install -r requirements.txt

# 1. (Optional) rebuild archetypes: open clusters/clusters.Rproj, run scrape.Rmd → shooters.Rmd → defense.Rmd

# 2. Extract shot features (downloads games; long-running)
python src/process_batch.py --start 0 --end 636 --output data/shot_features_full.csv

# 3. Centralized model
python src/train_xgboost.py
python src/tune_xgboost.py --n-iter 30

# 4. POE applications (write to results/)
python src/compute_poe.py
python src/compute_def_poe.py
python src/correlate_poe.py

# 5. Federated experiments (needs flwr; ~hours)
cd src/federated
pip install -e .
python run_seeds.py                 # all 4 configs × 5 seeds
python aggregate_seeds.py
python paired_bootstrap_ci.py       # from project root: python src/federated/paired_bootstrap_ci.py

# 6. Thesis tables/figures
python src/thesis_results.py
```

## Headline results (2015–16, game-disjoint test set)

| Model | Brier | Log loss | ROC-AUC |
|---|---|---|---|
| Constant (base rate) | 0.2475 | 0.6881 | 0.500 |
| Distance-only logistic | 0.2400 | 0.6728 | 0.603 |
| XGBoost (full features) | ~0.220 | ~0.628 | ~0.671 |
| Federated (4 configs, 5 seeds) | 0.2274–0.2279 | ~0.646 | 0.6446–0.6468 |

Federation costs ≈ +2.6% relative Brier vs the centralized baseline; team-level non-IID partitioning is not measurably worse than the IID control.

## Known issues / caveats

- **Legacy spacing feature.** `spatial.get_spacing_area` now returns true hull areas (`ConvexHull.volume`), but the shipped `shot_features_valid2.csv` was extracted with the old perimeter-based version (2-D `ConvexHull.area`), so its `ratio_off_def_hull` column is a perimeter ratio. Re-extract if the area semantics matter; the trained models are consistent with the shipped CSV.
- **Dataset-count mismatch with the thesis text.** The thesis data section quotes 630 games / 97,826 shots (from `shot_features_valid.csv`); the trained models actually use 613 games / 95,219 shots (`shot_features_valid2.csv`).
- **Stale result CSVs.** `results/headline_metrics.csv` and `results/per_zone_metrics.csv` come from a different run than the thesis Tables 4.1/4.2 (differences in the 3rd decimal). Re-run `thesis_results.py` to regenerate.
- **`compute_poe.py` must run before `compute_def_poe.py`/`correlate_poe.py`** — the latter two read `results/poe_per_shot.csv` / `results/poe_leaderboard.csv`.
