# modnet.pt

PyTorch rewrite of [MODNet](https://github.com/matgenix/modnet) (pinned
upstream v0.4.4, `1281977`) with a scikit-learn-style API: `fit(X, y)` on
plain DataFrames, no `MODData`. Full functional parity — matminer
featurization presets, FRR/NMI feature selection, joint-learning network,
preset + genetic hyperparameter search, bootstrap ensembles and
MC-dropout uncertainty.

```python
from modnet_pt import (
    MatminerFeaturizer, FRFeatureSelector, MODNetRegressor,
)

pipe = Pipeline([
    ("featurize", MatminerFeaturizer(preset="matminer_2023")),
    ("select",    FRFeatureSelector(n_features=64)),
    ("model",     MODNetRegressor(epochs=200)),
])
pipe.fit(df_structures, y)     # y = DataFrame, 1..n target columns
preds = pipe.predict(df_structures)
```

## Components

| Class | Role | Upstream port |
|---|---|---|
| `MatminerFeaturizer` | sklearn transformer; six presets (composition-only ×3, structure+site ×3) | `featurizers/featurizers.py`, `presets/` |
| `FRFeatureSelector` | NMI relevance + FRR redundancy ranking, dynamic p/c, multi-target merge, `selection_trace_` | `preprocessing.py` |
| `MODNetRegressor` | trunk → group blocks → heads, joint learning, weighted per-head loss | `models/vanilla.py` |
| `MODNetClassifier` | softmax (multiclass) / sigmoid (`multi_label=True`) heads, `predict_proba` | `models/vanilla.py:186-199` |

Both estimators are `clone`-safe and `Pipeline`-composable. `y` may be a
Series, ndarray, or DataFrame with several target columns; the `targets`
parameter groups columns into shared-head subgroups (singly- or
doubly-nested lists, or `"auto"`).

## Hyperparameter search

```python
MODNetRegressor(search="preset")   # upstream gen_presets grid + inner 5-fold
MODNetRegressor(search="ga",       # genetic algorithm (Individual genome)
                size_pop=20, num_generations=10, nested=5,
                early_stopping=4, refit=5)
```

`search_folds`/`nested` control inner CV folds (≤1 coerces to 5, as
upstream); `presets=` accepts a custom preset list. GA `refit>0` produces
a bootstrap ensemble of the winning genome; `refit=0` ensembles the
fold-models of the top-10 individuals. Diagnostics land in
`preset_search_`/`ga_results_`.

## Ensembles & uncertainty

```python
MODNetRegressor(bootstrap=8).fit(X, y)
mean = est.predict(X); std = est.predict_std(X)      # epistemic (ensemble)
m, s = est.predict_uncertainty(X, n_passes=32)      # MC-dropout
```

`dropout=p` (or `uncertainty=True` → p=0.1) enables MC-dropout layers;
`predict_uncertainty` freezes BatchNorm and runs stochastic passes.

## Validation utilities (`modnet_pt.validation`)

`learning_curve(est, X, y, sizes, cv)`, `nested_cv(est, X, y, outer, inner)`
(the 2021 paper's unbiased protocol), `matbench_cv(est, X, y)` — fixed
`MATBENCH_SEED` splits matching upstream `benchmark.py:16`.

## Status & parity

Milestones M0–M8 implemented (see `STATUS.md`). Algorithm ports cite
upstream file:line in docstrings. Statistical parity vs upstream on
`matbench_dielectric` is the remaining M8 item — the protocol is
`matbench_cv` with a `Pipeline(featurize → select → model)`; run it in the
compute env (TF upstream side) before claiming numbers.

License: MIT.
