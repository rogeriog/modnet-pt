# STATUS — modnet-pt

- **Phase:** M0–M8 implemented; remaining: matbench_dielectric statistical
  parity run (needs the compute env) + explainer Part-2 update
- **Last action:** 2026-11-07 — milestones M5–M7 + validation utilities:
  - **M5 `search="preset"`:** `gen_presets` grid ported (presets.py:12,
    with documented guards for tiny feature counts), inner k-fold
    validation on the upstream `MATBENCH_SEED` splits, per-candidate
    early stopping (train-loss monitor, patience 100, min_delta 1e-3 —
    vanilla.py:542-551), refit on full data with the winning preset.
    Scores: mean per-subgroup MAE (regression) / mean −OvR AUC
    (classification), matching `evaluate` (vanilla.py:768-839).
    `preset_search_` holds presets, score matrix, winner.
  - **M6 `search="ga"`:** full `Individual`/`FitGenetic` port — genome
    (act, loss, first-layer width, 3 fractions, xscale, impute, lr,
    batch, n_feat with upstream range rules), uniform crossover,
    mutation, `1/loss**5` parent weights, elitist ranking over all
    evaluated individuals, early stop on stagnant best, ensemble refit.
    Deviations (documented in search.py): serial eval, seeded RNG,
    upstream fraction-mutation bug fixed, widths clamped ≥1, weights
    shifted positive (upstream −AUC scores would crash `random.choices`).
  - **M7 ensembles/uncertainty:** `bootstrap=N` fits N bootstrap members
    (per-member seeds — upstream reuses seed 2943 so all resamples are
    identical, an upstream bug), `predict` = member mean,
    `predict_std` = member std. `predict_uncertainty(X, n_passes)` =
    MC-dropout (dropout param / `uncertainty=True`→0.1; BatchNorm frozen
    during passes) returning (mean, std). Ensemble members round-trip
    through save/load.
  - **`modnet_pt.validation`:** `learning_curve`, `nested_cv` (outer
    generalization × inner search — the paper's protocol),
    `matbench_cv` (fixed MATBENCH_SEED splits).
  - Tests: 80 fast + slow GA/joint/structure tests green; ruff clean.
- **Next step:** M8 remainder — matbench_dielectric parity table (run
  `matbench_cv` in the compute env vs pinned upstream numbers), explainer
  Part-2 API diff
- **Open:** upstream `matminer_2024_fast` preset exists in the ppdebreuck
  fork — candidate 7th preset, needs a DESIGN.md note if added
