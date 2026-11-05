# AGENTS.md — modnet-pt

Working agreement for agents working on the `modnet.pt` PyTorch rewrite.

## What this is

A fresh Python package — PyTorch reimplementation of MODNet with a
scikit-learn-style API that works directly on DataFrames. **No `MODData`.**
Sibling reference material lives in `../modnet-nutshell/` — read its
`research/code-map.md` before touching algorithms; it maps every upstream
file:line at the pinned commit `1281977` (v0.4.4).

## Ground rules

- **DESIGN.md is the contract.** No implementation before Rogério approves it.
  Any deviation from DESIGN.md gets discussed in chat first, then written back.
- **Parity target:** full functional parity with upstream MODNet —
  featurization presets, FRR/NMI feature selection, joint-learning network,
  GA hyperparameter search, ensembles/uncertainty, validation utilities —
  but with a PyTorch backend and a sklearn estimator contract.
- **sklearn contract is sacred:** `fit(X, y)` / `predict(X)` / `transform(X)`,
  `get_params`/`set_params`, clonable, composable in `Pipeline`. y may be a
  DataFrame with several target columns (joint learning).
- **Numerical honesty:** every algorithm port cites its upstream
  file:line in a docstring. Parity tests compare against the pinned upstream
  checkout on small seeded problems before we claim equivalence.
- **Upstream stays third-party:** the pinned clone lives in
  `../modnet-nutshell/research/modnet-src/` for reference and parity tests
  only. Never vendor-edit it; never import it at runtime.
- **Python:** package dir `modnet_pt/` (dots are illegal in import names —
  see DESIGN.md open questions for the naming decision).
- **Testing:** pytest + `sklearn.utils.estimator_checks` from day one;
  torch seeds pinned; no GPU required for CI.
- **Commits:** short imperative summaries.
- **Repo:** created by Rogério on GitHub (workspace rule — agents never
  create repos); after he creates it, run `bin/gh-setup.sh` from the
  workspace and send him the deploy key.

## Guardrails

- Don't reimplement what a maintained library provides (torch, sklearn,
  matminer are dependencies, not things to port).
- Don't silently drop features to make the port easier — flag them as
  TODO(parity) in DESIGN.md instead.
- Secrets never enter this repo.
