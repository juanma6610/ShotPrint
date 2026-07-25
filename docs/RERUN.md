# Full pipeline rerun (631-game dataset)

Run everything from the project root, inside the `mecasport` conda env:

```
conda activate mecasport
cd C:\Users\juanm\Documents\KUL_MAI\TFM\TFM-KUL-Juan
```

## Step 0 — make your 631-game CSV the canonical input

Every downstream script reads `data/shot_features_valid2.csv`. Point it at your new
extraction (rename or copy). The old 613-game file is overwritten on purpose.

```
copy /Y data\shot_features_631.csv data\shot_features_valid2.csv
```

Sanity check (should print ~95k–98k rows and 48 columns incl. the named archetypes):

```
python -c "import pandas as pd; d=pd.read_csv('data/shot_features_valid2.csv'); print(len(d),'rows', d['game_id'].nunique(),'games', len(d.columns),'cols'); print('makes', round(d['made_shot'].mean(),4))"
```

## Step 1 — centralized model

```
python src/tune_xgboost.py --n-iter 30      # -> data/xgb_shot_model_tuned.json, results/best_params.json, results/tuning_history.csv
python src/train_xgboost.py                 # -> data/xgb_shot_model.json + reliability/importance figures
```

## Step 2 — Points Over Expectation (order matters)

```
python src/compute_poe.py        # -> results/poe_per_shot.csv, results/poe_leaderboard.csv
python src/compute_def_poe.py    # needs results/poe_per_shot.csv -> DEF-POE leaderboard + matchup heatmap
python src/correlate_poe.py      # needs results/poe_leaderboard.csv -> POE vs advanced-metric correlations
```

## Step 3 — thesis tables & figures

```
python src/thesis_results.py     # -> results/headline_metrics.csv, per_zone_metrics.csv, poe_top_bottom_10, figures/*
python src/plot_per_zone_poe.py  # -> figures/per_zone_poe.png
```

## Step 4 — federated experiments

Runs 4 configs (bagging/cyclic x iid/team) x 5 seeds. The two cyclic configs do 1500
rounds each, so the **full sweep is multi-hour**. Do the install once.

```
cd src\federated
pip install -e .
python run_seeds.py                       # full sweep: 20 runs
python aggregate_seeds.py                 # -> results/federated/_aggregated_summary.csv (thesis Table 4.3)
cd ..\..
python src/federated/paired_bootstrap_ci.py   # needs data/xgb_shot_model_tuned.json + per-seed best models
```

Quick smoke test instead of the full sweep (one config, one seed):

```
cd src\federated
python run_seeds.py --only bagging_team --seeds 42
cd ..\..
```

## Step 5 — regenerate remaining thesis figures (optional)

The per-figure scripts in `figures_thesis/scripts/` write into `figures/`. Run the ones
you need, e.g.:

```
python figures_thesis/scripts/build_shap_beeswarm.py
python figures_thesis/scripts/build_lineup_leaderboards.py
python figures_thesis/scripts/build_xfg_scatter.py
```

## Notes

- Update the thesis counts to your new dataset (631 games / N shots) in the abstract,
  data section, and any table captions.
- `results/` and `figures/` are git-ignored (regenerated), so nothing here needs committing.
- If a script can't find the CSV, you skipped Step 0.
```
