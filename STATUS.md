# STATUS — modnet-pt

- **Phase:** M4 done — classifier + joint-learning parity experiment;
  M5 (preset search) is next
- **Last action:** 2026-11-07 — implemented milestone M4:
  - **`MODNetClassifier`:** multiclass softmax heads + cross-entropy,
    multi-label sigmoid heads (`multi_label=True`, exactly-2-classes
    enforced), joint classification over multiple class columns;
    `predict` returns labels, `predict_proba` returns a MultiIndexed
    DataFrame (target × class); early stopping via `val_fraction`/`patience`
    (best-val weights restored); save/load round-trips predictions.
  - **Joint-learning parity experiment** (`tests/test_joint.py`): the
    paper's vibrational dataset is unavailable offline, so targets are
    scaled noisy views of one latent (the multi-temperature structure).
    Calibrated result: **~10.4% joint-vs-single MAE gain at n=60,
    shrinking to ~3% at n=150** — the paper's pattern (~8%, larger at
    smaller sizes, p. 8). An earlier identical-target variant showed
    parity, not gain — do not cite it.
  - Tests: 56 fast + slow joint/structure tests; ruff clean.
- **Next step:** M5 — `search="preset"` grid (`gen_presets` port)
- **Open:** upstream `matminer_2024_fast` preset exists in the ppdebreuck
  fork — candidate 7th preset, needs a DESIGN.md note if added
