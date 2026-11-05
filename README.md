# modnet.pt

PyTorch rewrite of [MODNet](https://github.com/matgenix/modnet) with a
scikit-learn-style API: `fit(X, y)` on plain DataFrames, no `MODData`.
Full functional parity — featurization presets, FRR feature selection,
joint-learning network, GA hyperparameter search, ensembles/uncertainty.

- `DESIGN.md` — the contract (draft, awaiting Rogério's approval)
- `AGENTS.md` — working rules for agents
- Reference material: `../modnet-nutshell/research/` (pinned upstream v0.4.4,
  code map, both De Breuck papers)

Status: DESIGN approved (2026-11-05) — M0 scaffold in place; milestones M1–M8 per DESIGN.md. License: MIT.
