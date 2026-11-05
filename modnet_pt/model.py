"""PyTorch trunk + heads network for joint learning.

Port target (upstream v0.4.4 @ 1281977):
  modnet/models/vanilla.py:122  build_model  (shared trunk → group blocks → heads)
  modnet/models/vanilla.py:147  BatchNorm only in multi-target mode
  modnet/models/vanilla.py:388  min-max/impute semantics for inputs
  modnet/models/vanilla.py:462  per-head loss weighting
Architecture is deliberately small: four dense blocks, per-property heads.
See DESIGN.md §2.3 (milestones M3–M4).
"""

from __future__ import annotations

from torch import nn


class TrunkMultiHead(nn.Module):
    """Shared trunk, one intermediate block per target group, one head per property.

    Block layout mirrors vanilla.py:122 ``build_model``: dense block 1 (+
    BatchNorm when multi-target), one dense block per target group, then one
    output head per property in the group. Implementation: milestone M3.
    """

    def __init__(self, n_in: int, num_neurons=([64], [32], [16], [16]),
                 n_groups: int = 1, multi_target: bool = True,
                 act: str = "relu", out_act: str = "linear"):
        super().__init__()
        raise NotImplementedError("milestone M3 — see DESIGN.md §2.3")
