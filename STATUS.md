# STATUS — modnet-pt

- **Phase:** M3 done — core vertical slice implemented (featurizer →
  selector → regressor); M4 (joint-learning parity + classifier) is next
- **Last action:** 2026-11-05 — implemented milestones M1–M3 on top of the
  M0 scaffold (Rogério approved implementation):
  - **M1 `MatminerFeaturizer`:** all six presets (composition-only ×3,
    structure+site ×3), upstream `Featurizer|feature` naming, atomic-
    orbital/symmetry recoding, oxidation-state pipeline; fit learns
    all-NaN/duplicate drops on training data only (leak-free improvement
    over upstream `MODData.featurize`); `fit_transform` featurizes once.
  - **M2 `FRFeatureSelector`:** `nmi_target` / `get_cross_nmi` / FRR greedy
    ranking ported per-pair from `preprocessing.py` (bit-for-bit MI calls,
    same seeds); dynamic p/c + `rr_parameters` overrides; multi-target
    `merge_ranked`; `selection_trace_` per-step RR scores; joblib instead
    of raw multiprocessing.
  - **M3 `MODNetRegressor`:** torch trunk/group/head network
    (`vanilla.py:122` architecture, BatchNorm only when multi-target),
    upstream scaling/imputation semantics, weighted per-head loss,
    target hierarchy (singly- and doubly-nested accepted), DataFrame
    predictions with out-of-bounds remap, save/load, `random_state`
    reproducibility; sklearn pipeline composition works end-to-end.
  - Tests: 45 fast + 2 slow (structure featurization) green locally
    (py3.10, torch 2.14 CPU, matminer 0.9.3); ruff clean. FRR ranking
    parity test transcribes upstream `get_features_dyn` verbatim as
    reference.
- **Next step:** M4 — joint learning parity on the vibrational toy data
  (target groups, shared heads, the ~8% joint-vs-single gain) +
  `MODNetClassifier`
- **Open:** upstream `matminer_2024_fast` preset exists in the ppdebreuck
  fork — candidate 7th preset, needs a DESIGN.md note if added
