"""modnet.pt — PyTorch rewrite of MODNet with a scikit-learn-style DataFrame API.

Full functional parity with upstream MODNet (matgenix/modnet, pinned v0.4.4):
featurization presets, FRR/NMI feature selection, joint-learning network,
GA hyperparameter search, ensembles/uncertainty — without MODData.

Design contract: ../DESIGN.md (milestones M0–M8).
"""

__version__ = "0.1.0a0"

__all__ = [
    "MatminerFeaturizer",
    "FRFeatureSelector",
    "MODNetRegressor",
    "MODNetClassifier",
    "__version__",
]

from modnet_pt.estimator import MODNetClassifier, MODNetRegressor
from modnet_pt.featurizers import MatminerFeaturizer
from modnet_pt.selection import FRFeatureSelector
