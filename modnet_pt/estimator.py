"""sklearn estimator surface for the MODNet network.

Port target (upstream v0.4.4 @ 1281977):
  modnet/models/vanilla.py:48    MODNetModel.__init__ (defaults kept here)
  modnet/models/vanilla.py:266   fit (scaling, target groups, loss weights)
  modnet/models/ensemble.py:26   EnsembleMODNetModel   → bootstrap=  (M7)
  modnet/models/bayesian.py:290  predict uncertainty   → uncertainty=  (M7)
  modnet/model_presets/presets.py:12  gen_presets       → search="preset" (M5)
  modnet/hyper_opt/fit_genetic.py:15  Individual / FitGenetic → search="ga" (M6)

The sklearn contract is the product: y is a DataFrame with 1..n target columns
(joint learning), get_params/set_params/clone all work, predict returns a
target-named DataFrame. See DESIGN.md §2.3–2.4.
"""

from __future__ import annotations

from sklearn.base import BaseEstimator, RegressorMixin


class MODNetRegressor(RegressorMixin, BaseEstimator):
    """MODNet regressor with optional joint learning over target columns.

    Parameters (defaults mirror vanilla.py:48). fit/predict: milestones M3–M4;
    preset/GA searches: M5–M6; ensembles/uncertainty: M7.
    """

    def __init__(self, targets=None, weights=None, n_feat: int = 64,
                 num_neurons=([64], [32], [16], [16]), act: str = "relu",
                 out_act: str = "linear", lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 128, xscale: str = "minmax",
                 impute_missing=0, loss: str = "mse", search: str = "none",
                 bootstrap: int = 0, uncertainty: bool = False,
                 random_state: int | None = None, device: str = "cpu"):
        self.targets = targets
        self.weights = weights
        self.n_feat = n_feat
        self.num_neurons = num_neurons
        self.act = act
        self.out_act = out_act
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.xscale = xscale
        self.impute_missing = impute_missing
        self.loss = loss
        self.search = search
        self.bootstrap = bootstrap
        self.uncertainty = uncertainty
        self.random_state = random_state
        self.device = device

    def fit(self, X, y):
        raise NotImplementedError("milestones M3–M4 — see DESIGN.md §2.3")

    def predict(self, X):
        raise NotImplementedError("milestones M3–M4 — see DESIGN.md §2.3")


class MODNetClassifier(BaseEstimator):
    """Categorical-head variant (softmax/sigmoid, cross-entropy). M4+."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("milestone M4 — see DESIGN.md §2.3")

    def predict(self, X):
        raise NotImplementedError("milestone M4 — see DESIGN.md §2.3")
