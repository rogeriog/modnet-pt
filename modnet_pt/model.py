"""PyTorch trunk + heads network for joint learning.

Port of ``MODNetModel.build_model`` (upstream vanilla.py:122-220):
  - shared trunk: dense block 1 (+ BatchNorm1d only in multi-target mode,
    vanilla.py:147) followed by one dense block 2 per target group;
  - one dense block 3 + output head per property subgroup, where a
    subgroup shares a single multi-output head (vanilla.py:177-200);
  - the 4th entry of ``num_neurons`` is accepted but unused, exactly as
    upstream ``build_model`` only consumes blocks 1-3.
See DESIGN.md §2.3 (milestones M3–M4).
"""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "linear": nn.Identity,
}


def _activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[name]()


class TrunkMultiHead(nn.Module):
    """Shared trunk, one intermediate block per target group, one head per
    property subgroup.

    ``targets`` is the doubly-nested upstream hierarchy
    ``[[[prop, ...], [prop, ...]], ...]``: one block-2 per outer group, one
    head per subgroup, head width = ``len(subgroup)`` (regression). For
    classification subgroups (``num_classes[name] >= 2``) the head has one
    output per class with a softmax activation, mirroring vanilla.py:186-199.
    """

    def __init__(
        self,
        n_in: int,
        targets,
        num_neurons=([64], [32], [16], [16]),
        num_classes: dict | None = None,
        act: str = "relu",
        out_act: str = "linear",
    ):
        super().__init__()
        num_classes = num_classes or {}

        flat = [prop for group in targets for sub in group for prop in sub]
        multi_target = len(flat) > 1

        # Block 1: shared trunk
        trunk = []
        dim = n_in
        for width in num_neurons[0]:
            trunk.append(nn.Linear(dim, width))
            trunk.append(_activation(act))
            if multi_target:
                trunk.append(nn.BatchNorm1d(width))
            dim = width
        self.trunk = nn.Sequential(*trunk)

        # Block 2: one per target group
        self.group_blocks = nn.ModuleList()
        for _ in targets:
            block = []
            d = dim
            for width in num_neurons[1]:
                block.append(nn.Linear(d, width))
                block.append(_activation(act))
                if multi_target:
                    block.append(nn.BatchNorm1d(width))
                d = width
            self.group_blocks.append(nn.Sequential(*block))

        # Block 3 + heads: one per property subgroup
        self.heads = nn.ModuleList()
        self._head_meta = []
        for group_idx, group in enumerate(targets):
            for sub in group:
                n_classes = num_classes.get(sub[0], 0)
                block = []
                d = num_neurons[1][-1] if num_neurons[1] else dim
                for width in num_neurons[2]:
                    block.append(nn.Linear(d, width))
                    block.append(_activation(act))
                    if multi_target:
                        block.append(nn.BatchNorm1d(width))
                    d = width
                if n_classes >= 2:
                    head = nn.Linear(d, n_classes)
                    self._head_meta.append({"name": sub[0], "width": n_classes, "classes": n_classes})
                else:
                    head = nn.Linear(d, len(sub))
                    self._head_meta.append(
                        {"name": sub[0], "width": len(sub), "classes": 0}
                    )
                self.heads.append(nn.Sequential(*block, head))

        self.out_act = _activation(out_act)
        self.targets = targets
        self.multi_target = multi_target

    @property
    def head_names(self):
        """Head names in forward order (one per subgroup, upstream
        ``output_names``, vanilla.py:203)."""
        return [meta["name"] for meta in self._head_meta]

    @property
    def head_widths(self):
        return [meta["width"] for meta in self._head_meta]

    def forward(self, x: torch.Tensor):
        shared = self.trunk(x)
        outputs = []
        head_idx = 0
        for group_idx, group in enumerate(self.targets):
            group_out = self.group_blocks[group_idx](shared)
            for _ in group:
                outputs.append(self.heads[head_idx](group_out))
                head_idx += 1
        regression = all(meta["classes"] == 0 for meta in self._head_meta)
        if regression:
            outputs = [self.out_act(out) for out in outputs]
        # classification heads return raw logits: loss functions
        # (CrossEntropy / BCEWithLogits) expect them, and the estimator
        # applies softmax/sigmoid at predict time
        return outputs
