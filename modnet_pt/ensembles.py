"""Bootstrap ensembles and MC-dropout uncertainty.

Port targets (upstream v0.4.4 @ 1281977):
  modnet/models/ensemble.py:26   EnsembleMODNetModel (mean prediction, std)
  modnet/models/bayesian.py:290  MC-dropout predict
Contract per De Breuck 2021 benchmarking paper pp. 4–5: mean = prediction,
std = epistemic uncertainty. See DESIGN.md §2.3 (milestone M7).
"""

from __future__ import annotations

raise NotImplementedError("milestone M7 — see DESIGN.md §2.3")
