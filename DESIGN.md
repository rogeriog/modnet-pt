# DESIGN.md — modnet.pt (PyTorch rewrite of MODNet)

**Status: DRAFT — waiting on Rogério's approval before any implementation.**
Reference: `../modnet-nutshell/research/code-map.md` (upstream pinned at
`1281977`, v0.4.4) and the two De Breuck papers.

---

## 1. The pitch

MODNet is a strong small-data materials ML framework trapped behind a Keras-era
API: everything funnels through a `MODData` container, and the sklearn layer
upstream ships half-finished (`sklearn.py` — `RR` works, `MODNetFeaturizer.transform`
and the `MODNet` estimator are stubs). **`modnet.pt` finishes what upstream
started:** a PyTorch backend and a clean sklearn contract that works directly
on DataFrames — no wrapper ceremony, composable in `Pipeline`, CV-friendly.

```python
from modnet_pt import MatminerFeaturizer, FRFeatureSelector, MODNetRegressor

pipe = Pipeline([
    ("featurize", MatminerFeaturizer(preset="matminer_2023")),
    ("select",    FRFeatureSelector(n_features=64)),
    ("model",     MODNetRegressor(n_neurons=([64],[32],[16],[16]), epochs=200)),
])
pipe.fit(df_structures, y)        # y = DataFrame with 1..n target columns
preds = pipe.predict(df_structures)
```

## 2. API surface (sklearn contract)

### 2.1 Featurization — `modnet_pt.featurizers`

```python
MatminerFeaturizer(preset="composition_only_matminer_2023" | "matminer_2023" | ...)
```

- sklearn `TransformerMixin`/`BaseEstimator`. Input: DataFrame with a
  `"composition"` (pymatgen `Composition`) or `"structure"` column.
- `fit` learns the all-NaN / duplicate-column drops **on the training set only**
  (upstream `MODData.featurize` does this implicitly inside `feature_selection`;
  here the transformer owns it so pipelines stay leak-free).
- `transform` returns the featurized DataFrame (feature names preserved for
  the selector and for interpretability).
- Ported from `modnet/featurizers/featurizers.py:17-293` + presets
  (`presets/__init__.py:13`, six presets).

### 2.2 Feature selection — `modnet_pt.selection`

```python
FRFeatureSelector(n_features=64, rr_parameters=None, n_neighbors=3,
                  n_samples=6000, drop_thr=0.2)
```

- sklearn transformer. `fit(X, y)`: computes target NMI (`nmi_target`),
  cross-NMI (`get_cross_nmi`), then the FRR greedy ranking with the dynamic
  paper parameters `p = max(0.1, 4.5 − n^0.4)`, `c = 1e−6·n³`.
- `transform(X)`: keeps the top `n_features` in **ranked order**
  (order matters — it is what upstream calls "optimal descriptors").
- Exposes `ranking_`, `nmi_target_`, `cross_nmi_`, `selection_trace_`
  (per-step RR scores — the explainer's FRR widget replays exactly this).
- Multi-target: computes one ranking per target and merges with
  `merge_ranked` semantics (`preprocessing.py:516`).
- Ported from `preprocessing.py:45` (`compute_mi`), `:68` (`nmi_target`),
  `:175` (`get_cross_nmi`), `:297/:311` (dynamic p, c), `:325`
  (`get_features_relevance_redundancy`), `:476` (`get_features_dyn`).
- Keeps the sklearn-compat `RR` behaviour of upstream `sklearn.py:62` as the
  same class, finished properly.

### 2.3 Model — `modnet_pt.model`

```python
MODNetRegressor(targets=None, weights=None, n_feat=64,
                num_neurons=([64], [32], [16], [16]),
                act="relu", out_act="linear",
                lr=1e-3, epochs=200, batch_size=128,
                xscale="minmax", impute_missing=0,
                loss="mse", device="cpu|cuda")
MODNetClassifier(...)  # softmax/sigmoid heads, cross-entropy
```

- `fit(X, y)`: `X` DataFrame/ndarray (already featurized + selected, or raw —
  inside a `Pipeline` the upstream steps do it); `y` DataFrame/ndarray with
  one column per target.
- **Joint learning** (upstream `models/vanilla.py:122` `build_model`):
  - targets passed as a **nested list of groups** (e.g.
    `[["S_300K","S_400K"], ["F_300K"]]`) or auto-grouped
    (`targets="auto"` → one group per column).
  - Architecture: shared trunk (block 1 + BatchNorm) → one intermediate block
    per group → one head per property. BatchNorm appears only in multi-target
    mode, exactly as upstream (`vanilla.py:147/:162/:177`).
  - Loss = weighted sum over heads (`loss_weights=self.weights`,
    `vanilla.py:462-467`).
- PyTorch module: `nn.Linear`/`nn.BatchNorm1d` per block; Adam
  (`torch.optim.Adam`); MAE or MSE loss per head, weighted by `weights`.
- Scaling/imputation follow upstream `fit` (`vanilla.py:388-403`):
  min-max to [−0.5, 0.5] with NaN → −1, or standard scaling; stored in the
  estimator so `predict` transforms identically.
- `predict(X)` returns a DataFrame with target names as columns
  (classification: class probabilities / labels).
- `get_params`/`set_params` cover everything; `clone`-safe; torch seed pinned
  via `random_state` (sklearn convention).
- Save/load: `save(path)` → torch state_dict + a JSON of all sklearn params
  + target/feature metadata; `load(path)` classmethod restores.
- Upstream `EnsembleMODNetModel` (`models/ensemble.py:26`) and
  `BayesianMODNetModel` (`models/bayesian.py:29`) map to:
  - `MODNetRegressor(bootstrap=8)` → fits 8 models on bootstrap resamples,
    `predict` returns mean with `predict_std()` for epistemic uncertainty
    (mean/std contract from the 2021 paper, p. 4–5).
  - MC-dropout uncertainty: `predict_uncertainty(X, n_passes=32)` mirroring
    `bayesian.py:290`.

### 2.4 Hyperparameter search

- `fit_preset(data)` equivalent: `MODNetRegressor(search="preset")` runs the
  upstream grid (`model_presets/presets.py:12` `gen_presets` — batch sizes,
  lr, feature counts, three architectures) with the same inner-validation
  rule, then refits on the full training set.
- **GA search** (upstream `hyper_opt/fit_genetic.py`), ported last:
  - `MODNetRegressor(search="ga", size_pop=20, num_generations=10, nested=5,
    early_stopping=4, refit=5)`.
  - Genome = upstream `Individual` (`fit_genetic.py:15`): activation, loss,
    first-layer neurons, three data fractions, scaler, impute strategy,
    lr, batch size, n_feat.
  - Fitness = nested-CV MAE; selection weights `1/loss**5`; crossover +
    mutation; elitist merge; early stopping; winner refit into an ensemble
    (`run` at `:510`).
- Both searches follow sklearn optics: they are `search` modes on the
  estimator, not separate classes.

### 2.5 Validation utilities — `modnet_pt.validation`

- `learning_curve(estimator, X, y, sizes, cv)` — parity/learning-curve data.
- `nested_cv(estimator, X, y, outer=5, inner=5)` — the paper's unbiased
  protocol (2021 benchmarking paper, p. 6).
- MatBench harness: `matbench(task)` runs the MatBench protocol
  (`modnet/matbench/benchmark.py` equivalent).

## 3. What we do NOT port

- `MODData` — gone. Its two real jobs (featurize + hold selected features)
  become transformer state.
- Keras checkpoints / `.h5` pretrained models — not loadable; pretrained
  models are a stretch goal retrained in torch.
- The `db_file` featurization cache — replaced by the transformer's
  fit/transform split (sklearn caching covers it).
- `DeprecatedMODNetModel` (`vanilla.py:1056`) — legacy compat, dropped.

## 4. Parity & testing plan

1. **Algorithm parity:** NMI/FRR selection on a fixed synthetic matrix must
   reproduce the pinned upstream ranking bit-for-bit (same sklearn
   `mutual_info_regression`, `n_neighbors=3`). Architecture defaults and
   scaling semantics match `vanilla.py`; joint-vs-single loss weighting
   matches `:462-467`.
2. **Statistical parity:** on `matbench_dielectric` (the explainer's dataset),
   a parity table vs pinned upstream — same protocol (outer 5-fold, inner 5),
   report MAE ± spread. Target: within noise of upstream, not identical
   (torch/Keras numerics differ).
3. **sklearn checks:** `estimator_checks` for every public class; pipeline
   composition tests; clone/repr tests.
4. **GA parity:** same genome, same fitness rules; assert the GA finds ≥ the
   grid-search optimum on a toy problem (mirrors `tests/test_hyper_opt.py`).

## 5. Module map

```
modnet_pt/
├── featurizers.py     # MatminerFeaturizer (TransformerMixin) + presets
├── selection.py       # FRFeatureSelector (NMI + FRR), selection_trace_
├── model.py           # torch nn.Module: trunk + group blocks + heads
├── estimator.py       # MODNetRegressor / MODNetClassifier (sklearn glue)
├── scaling.py         # minmax/impute + standard scaler (upstream semantics)
├── search.py          # preset grid + GA (Individual/FitGenetic port)
├── ensembles.py       # bootstrap ensemble + MC-dropout uncertainty
├── validation.py      # learning curves, nested CV, matbench harness
└── presets/           # ranked feature lists as data artifacts
```

Dependencies: `torch`, `scikit-learn`, `matminer`, `pymatgen`, `pandas`,
`numpy`. Python ≥ 3.10 (torch floor). CPU-first; CUDA optional.

## 6. Milestones

- **M0** scaffold + CI + sklearn-check harness (this repo).
- **M1** `MatminerFeaturizer` with the six presets; parity vs upstream counts.
- **M2** `FRFeatureSelector`: NMI, cross-NMI, FRR ranking + trace.
- **M3** `MODNetRegressor` single-target: fit/predict/evaluate, scaling,
  save/load.
- **M4** joint learning: target groups, shared trunk, loss weights; the
  ~8 % joint-vs-single gain reproducible on the vibrational toy data.
- **M5** `search="preset"` grid.
- **M6** GA search (full `Individual`/`FitGenetic` port).
- **M7** bootstrap ensemble + MC-dropout uncertainty.
- **M8** docs, `matbench_dielectric` parity table, explainer §7 update with
  the old-vs-new API diff.

## 7. Open questions for Rogério

1. **Naming:** GitHub repo + PyPI. PyPI forbids dots → `modnet-pt` with
   import name `modnet_pt` is the clean reading of "modnet.pt". OK, or do you
   want the repo literally named `modnet.pt` (GitHub allows it) with PyPI
   name `modnet-pt`?
2. **License:** MIT recommended (upstream is MIT). Your call.
3. **Featurization scope:** full featurization in `modnet.pt` (depends on
   matminer), or should featurization stay optional (extra) so the model
   install stays light?
4. **Torch version floor:** 2.x assumed.
5. **GPU:** CPU-first OK? (torch makes CUDA automatic if present.)
6. **Repo hosting:** per workspace rules you create the GitHub repo —
   name + visibility, then I'll wire keys and CI.
