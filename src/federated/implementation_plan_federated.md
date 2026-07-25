# Implementation Plan: Federated XGBoost with Flower

## Background

Your `shot_features_full.csv` is ready:
- **97,826 shots** × 44 columns
- **30 teams** (NBA 2015-16), well-balanced: ~3,260 shots/team (min 2,801, max 3,539)
- **630 games** with `game_id` — enables temporal analysis
- `team_id` and `game_id` already present as metadata columns ✅

No data re-extraction is needed. We build directly on top of `train_xgboost.py`.

## Decisions Summary

| Parameter | Value |
|-----------|-------|
| Partitioning | By `team_id` (30 clients) — primary; IID as control |
| Aggregation | `FedXgbBagging` (parallel, tree concatenation) |
| FL Rounds | 50 rounds |
| Trees per client per round | 1 |
| Total trees in global model | 50 × 30 = 1,500 |
| Mode | Simulation (`flwr run`) |
| Personalization | Discussion only (not implemented) |

---

## Proposed Project Structure

```
src/
└── federated/
    ├── pyproject.toml          # Flower project config (num clients, rounds, params)
    └── nba_federated/
        ├── __init__.py
        ├── task.py             # Data loading, partitioning, DMatrix helpers
        ├── client_app.py       # Flower XGBoost ClientApp (local training)
        └── server_app.py       # Flower ServerApp with FedXgbBagging
```

---

## Proposed Changes

### Federated Module

#### [NEW] `src/federated/nba_federated/task.py`

Handles all data logic. Three partitioning strategies:

```python
def load_partition(partition_id: int, strategy: str = "team") -> tuple:
    """
    Load local training data for one federated client.
    
    strategy:
      "team"      — partition by team_id (30 clients, Non-IID) [PRIMARY]
      "iid"       — stratified random split (30 clients, IID) [CONTROL]
    
    Returns: (X_train, X_test, y_train, y_test, scale_pos_weight)
    """
```

Key implementation detail: `train_xgboost.py` already defines `metadata_cols` and the feature columns. `task.py` reuses that exact same logic to stay consistent — same features, same class weight calculation, just scoped to one team's rows.

The returned data will be wrapped in `xgb.DMatrix` with the same hyperparameters your model uses today (`max_depth=5`, `eta=0.02`, `subsample=0.8`, etc.).

#### [NEW] `src/federated/nba_federated/client_app.py`

Each client = one NBA team. On each FL round:
1. Receives the current global model (serialized as bytes) from the server
2. Calls `xgb.train(..., num_boost_round=1, xgb_model=global_model)` — adds **exactly 1 tree** on top of the global model using local team data
3. Sends the updated model (global trees + 1 new local tree) back to server

```python
class XGBClient(fl.client.Client):
    def fit(self, parameters, config):
        # Deserialize global model
        # Train 1 new tree on local data
        # Return updated model bytes + num local samples
        ...

    def evaluate(self, parameters, config):
        # Evaluate global model on local test split
        # Return loss, num samples, {"auc": auc, "accuracy": acc}
        ...
```

#### [NEW] `src/federated/nba_federated/server_app.py`

Uses Flower's built-in `FedXgbBagging` strategy. Server logic:
- Collects trees from all 30 clients each round
- **Concatenates** them into the growing global ensemble
- Runs centralized evaluation on a held-out global test set after each round
- Logs global AUC, accuracy, and Brier score per round → enables convergence curve plot

```python
strategy = FedXgbBagging(
    fraction_fit=1.0,            # All 30 teams participate each round
    min_fit_clients=30,
    min_available_clients=30,
    evaluate_function=evaluate_global,   # Centralized eval on holdout set
)
```

#### [NEW] `src/federated/pyproject.toml`

Flower project configuration:

```toml
[tool.flwr.app]
publisher = "nba-shot-prediction"

[tool.flwr.app.components]
serverapp = "nba_federated.server_app:app"
clientapp = "nba_federated.client_app:app"

[tool.flwr.app.config]
num-server-rounds = 50
num-clients = 30
strategy = "team"           # "team" or "iid"
data-path = "../../data/shot_features_full.csv"
local-epochs = 1            # trees per round per client
params.max_depth = 5
params.eta = 0.02
params.subsample = 0.8
params.colsample_bytree = 0.8
params.objective = "binary:logistic"
params.eval_metric = "logloss"
params.tree_method = "hist"

[tool.flwr.federations.local-simulation]
options.num-supernodes = 30
```

Run with: `flwr run . local-simulation --stream`

---

## Experiments & Evaluation

Three experiments to run and compare in the thesis:

| Experiment | Script | Description |
|------------|--------|-------------|
| **E1: Centralized** | `train_xgboost.py` (existing) | All 97K shots pooled — upper bound |
| **E2: Federated IID** | `flwr run` with `strategy="iid"` | Control — random 30-way split |
| **E3: Federated Team** | `flwr run` with `strategy="team"` | Main experiment — 30 teams |

### Metrics to collect per experiment
- Final global **AUC**, **Accuracy**, **Brier Score** (on same held-out test set)
- **Convergence curve**: global AUC vs. FL round number
- **Per-client AUC**: how well the global model performs on each individual team
- **Privacy cost**: gap between E1 and E3 AUC

### Plots to generate
1. Convergence: AUC vs FL round (E2 vs E3 on same axes)
2. Bar chart: per-team AUC across all 30 teams
3. Summary table: E1 vs E2 vs E3 final metrics

---

## Verification Plan

### Before running FL
- `task.py` unit test: verify all 30 partitions have correct sizes, no data leakage between train/test splits, same feature columns as `train_xgboost.py`

### FL simulation run
```bash
cd src/federated
pip install flwr xgboost scikit-learn pandas
flwr run . local-simulation --stream
```
Expected: 50 rounds of logs, global AUC improving round-by-round, final AUC within ~2-3% of centralized baseline (E1).

### Sanity checks
- Global model after 50 rounds has exactly 1,500 trees (`model.num_boosted_rounds() == 1500`)
- Per-team evaluation: no single team below 55% AUC (would indicate a partitioning bug)
- IID (E2) AUC ≥ Team-based (E3) AUC — non-IID hurts, as expected

> [!NOTE]
> Personalization (Global + per-team fine-tuning) is intentionally left as a **thesis discussion point**. The architecture for it is straightforward: load the E3 global model as `xgb_model` and run additional local `xgb.train()` rounds — but implementing and evaluating it is optional scope.
