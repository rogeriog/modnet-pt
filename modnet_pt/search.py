"""Hyperparameter search: preset grid + genetic algorithm.

Port targets (upstream v0.4.4 @ 1281977):
  modnet/model_presets/presets.py:12   gen_presets grid
  modnet/hyper_opt/fit_genetic.py:15   Individual genome (act, loss, first-layer
                                       neurons, three fractions, scaler, impute,
                                       lr, batch, n_feat)
  modnet/hyper_opt/fit_genetic.py:85   crossover
  modnet/hyper_opt/fit_genetic.py:112  mutation
  modnet/hyper_opt/fit_genetic.py:510  FitGenetic.run (weighted 1/loss**5 parent
                                       selection, elitism, early stopping, refit)
See DESIGN.md §2.4 (milestones M5–M6).
"""

from __future__ import annotations

raise NotImplementedError("milestones M5–M6 — see DESIGN.md §2.4")
