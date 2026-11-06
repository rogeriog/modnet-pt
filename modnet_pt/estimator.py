"""sklearn estimator surface for the MODNet network.

Port of ``MODNetModel`` (upstream vanilla.py:48-467) onto the sklearn
contract: ``fit(X, y)`` / ``predict(X)``, ``get_params``/``set_params``,
clone-safe, y a DataFrame with 1..n target columns (joint learning).
Search modes (M5–M6), bootstrap ensembles and uncertainty (M7) raise
``NotImplementedError`` until their milestones land. See DESIGN.md §2.3–2.4.

The doubly-nested upstream ``targets`` hierarchy is preserved (one block-2
per outer group, one shared head per subgroup, vanilla.py:122), with a
usability improvement: singly-nested lists (``[["a", "b"], ["c"]]``, i.e.
one head per list) and ``None``/``"auto"`` (one head per target column) are
accepted and normalized to the upstream form.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from modnet_pt.model import TrunkMultiHead
from modnet_pt.scaling import ScaleImpute

_LOSSES = ("mse", "mae")


def _normalize_targets(targets, columns):
    """Normalize the ``targets`` spec to the doubly-nested upstream form
    ``[[[prop, ...], ...], ...]`` (vanilla.py:122 ``targets``)."""
    if targets is None or targets == "auto":
        # one head per target column, all sharing a single outer group
        return [[[c] for c in columns]]

    if not targets or not all(isinstance(t, (list, tuple)) for t in targets):
        raise ValueError(
            "targets must be None, 'auto', a nested list of property groups, "
            "or the upstream doubly-nested list."
        )
    if all(isinstance(t, str) for sub in targets for t in sub):
        # singly-nested: [["a", "b"], ["c"]] → one shared head per inner list
        return [[list(sub)] for sub in targets]

    for group in targets:
        if not all(isinstance(sub, (list, tuple)) for sub in group):
            raise ValueError(
                "targets must be nested 2 levels deep (property names inside "
                "subgroups inside groups)."
            )
    return [[list(sub) for sub in group] for group in targets]


class MODNetRegressor(RegressorMixin, BaseEstimator):
    """MODNet regressor with joint learning over target columns.

    Parameters (defaults mirror vanilla.py:48 and its ``fit`` defaults,
    vanilla.py:266): ``num_neurons`` blocks 1-3 build the trunk/group/head
    network, ``xscale``/``impute_missing`` follow upstream scaling semantics,
    ``targets`` accepts ``None`` (one head per column), a singly-nested list
    of shared-head groups, or the upstream doubly-nested hierarchy, and
    ``weights`` maps target names to per-head loss weights (vanilla.py:462).
    """

    def __init__(self, targets=None, weights=None, n_feat: int = 64,
                 num_neurons=([64], [32], [16], [16]), act: str = "relu",
                 out_act: str = "linear", lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 128, xscale: str = "minmax",
                 impute_missing=0, loss: str = "mse", search: str = "none",
                 bootstrap: int = 0, uncertainty: bool = False,
                 val_fraction: float = 0.0,
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
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.device = device

    def _check_config(self):
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {self.loss!r}")
        if self.search != "none":
            raise NotImplementedError(
                'search="preset"/"ga" arrives in milestones M5-M6 (DESIGN.md §2.4)'
            )
        if self.bootstrap:
            raise NotImplementedError(
                "bootstrap ensembles arrive in milestone M7 (DESIGN.md §2.3)"
            )
        if self.uncertainty:
            raise NotImplementedError(
                "uncertainty estimates arrive in milestone M7 (DESIGN.md §2.3)"
            )

    def _prepare_y(self, y):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(y, pd.DataFrame):
            return y, list(y.columns)
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = [f"y{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=names), names

    def fit(self, X, y):
        self._check_config()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            X = pd.DataFrame(np.asarray(X))

        y_frame, y_columns = self._prepare_y(y)
        if len(X) != len(y_frame):
            raise ValueError("X and y have different numbers of samples.")

        targets = _normalize_targets(self.targets, y_columns)
        flat = [p for group in targets for sub in group for p in sub]
        missing = [c for c in flat if c not in y_frame.columns]
        if missing:
            raise ValueError(f"targets not present in y: {missing}")
        self.targets_ = targets
        self.target_names_ = flat

        if self.n_feat > X.shape[1]:
            raise ValueError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {X.shape[1]}"
            )
        self.feature_names_used_ = list(X.columns)[: self.n_feat]
        x = X[self.feature_names_used_].values.astype(np.float64)

        weights = self.weights or {}
        head_weights = [
            float(weights.get(sub[0], 1))
            for group in targets
            for sub in group
        ]

        # Scale and impute inputs, then fit the network (vanilla.py:388-470)
        self.scale_impute_ = ScaleImpute(
            xscale=self.xscale, impute_missing=self.impute_missing
        ).fit(x)
        x = self.scale_impute_.transform(x)

        groups = [sub for group in targets for sub in group]
        y_list = [
            torch.tensor(
                y_frame[sub].values.astype(np.float64), dtype=torch.float64
            )
            for sub in groups
        ]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)

        n_train = len(x)
        if self.val_fraction and self.val_fraction > 0:
            idx = rng.permutation(n_train)
            n_val = max(1, int(self.val_fraction * n_train))
            val_idx, train_idx = idx[:n_val], idx[n_val:]
        else:
            val_idx, train_idx = np.array([], dtype=int), np.arange(n_train)

        self.network_ = TrunkMultiHead(
            self.n_feat, targets, num_neurons=self.num_neurons,
            act=self.act, out_act=self.out_act,
        ).double()
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        if self.loss == "mse":
            criterion = torch.nn.functional.mse_loss
        else:
            criterion = torch.nn.functional.l1_loss

        x_all = torch.tensor(x, dtype=torch.float64)
        self.history_ = {"loss": []}
        if len(val_idx):
            self.history_["val_loss"] = []

        self.network_.train()
        for _ in range(self.epochs):
            perm = rng.permutation(len(train_idx))
            losses = []
            for start in range(0, len(perm), self.batch_size):
                batch = train_idx[perm[start : start + self.batch_size]]
                xb = x_all[torch.tensor(batch)]
                yb = [y_i[torch.tensor(batch)] for y_i in y_list]
                optimizer.zero_grad()
                preds = self.network_(xb)
                total = torch.tensor(0.0, dtype=torch.float64)
                for pred, target, w in zip(preds, yb, head_weights):
                    total = total + w * criterion(pred, target)
                total.backward()
                optimizer.step()
                losses.append(total.detach().item())
            self.history_["loss"].append(float(np.mean(losses)))
            if len(val_idx):
                self.network_.eval()
                with torch.no_grad():
                    preds = self.network_(x_all[torch.tensor(val_idx)])
                    total = torch.tensor(0.0, dtype=torch.float64)
                    for pred, target, w in zip(
                        preds, [y_i[torch.tensor(val_idx)] for y_i in y_list],
                        head_weights,
                    ):
                        total = total + w * criterion(pred, target)
                self.history_["val_loss"].append(total.detach().item())
                self.network_.train()

        # Postprocessing bounds (vanilla.py:444-449)
        self.min_y_ = [y_frame[sub].values.min(axis=0) for sub in groups]
        self.max_y_ = [y_frame[sub].values.max(axis=0) for sub in groups]

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X, remap_out_of_bounds: bool = True):
        """Predict target values for X.

        Returns a DataFrame with one column per target (joint-learning
        subgroups expand to their member columns), mirroring upstream
        ``MODNetModel.predict`` (vanilla.py:696-760) including the
        out-of-bounds remapping to the training range.
        """
        check_is_fitted(self, "network_")
        if isinstance(X, pd.DataFrame):
            missing = [
                c for c in self.feature_names_used_ if c not in X.columns
            ]
            if missing:
                raise ValueError(f"predict input missing features: {missing[:3]}")
            x = X[self.feature_names_used_].values.astype(np.float64)
        else:
            arr = np.asarray(X)
            if arr.shape[1] != len(self.feature_names_used_):
                raise ValueError(
                    f"expected {len(self.feature_names_used_)} features, "
                    f"got {arr.shape[1]}"
                )
            x = arr.astype(np.float64)
        x = np.where(np.isfinite(x), x, np.nan)
        x = self.scale_impute_.transform(x)

        self.network_.eval()
        with torch.no_grad():
            preds = self.network_(torch.tensor(x, dtype=torch.float64))
        preds = [p.numpy() for p in preds]

        if remap_out_of_bounds:
            rng = np.random.default_rng(self.random_state)
            for i, vals in enumerate(preds):
                yrange = self.max_y_[i] - self.min_y_[i]
                upper = self.max_y_[i] + 0.25 * yrange
                lower = self.min_y_[i] - 0.25 * yrange
                for j in range(vals.shape[1]):
                    out = (vals[:, j] < lower[j]) | (vals[:, j] > upper[j])
                    vals[out, j] = (
                        rng.uniform(0, 1, size=out.sum()) * yrange[j]
                        + self.min_y_[i][j]
                    )

        columns = [p for group in self.targets_ for sub in group for p in sub]
        out = np.hstack([p.reshape(len(p), -1) for p in preds])
        return pd.DataFrame(out, columns=columns)

    def save(self, path: str):
        """Save fitted model: torch state_dict + metadata JSON.

        Mirrors DESIGN.md §2.3: the sklearn params, target hierarchy,
        feature names and scaling state round-trip through ``load``."""
        check_is_fitted(self, "network_")
        meta = {
            "params": self.get_params(),
            "targets": self.targets_,
            "target_names": self.target_names_,
            "feature_names_used": self.feature_names_used_,
            "min_y": [m.tolist() for m in self.min_y_],
            "max_y": [m.tolist() for m in self.max_y_],
        }
        torch.save(
            {
                "state_dict": self.network_.state_dict(),
                "meta_json": json.dumps(meta),
                "scale_impute": self.scale_impute_,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> MODNetRegressor:
        """Restore a model saved with ``save``."""
        blob = torch.load(path, weights_only=False)
        meta = json.loads(blob["meta_json"])
        params = meta["params"]
        # JSON turns the num_neurons tuples into lists; restore the
        # constructor's tuple-of-lists form so get_params round-trips
        params["num_neurons"] = tuple(
            list(block) for block in params["num_neurons"]
        )
        est = cls(**params)
        est.targets_ = [[list(sub) for sub in group] for group in meta["targets"]]
        est.target_names_ = meta["target_names"]
        est.feature_names_used_ = meta["feature_names_used"]
        est.min_y_ = [np.array(m) for m in meta["min_y"]]
        est.max_y_ = [np.array(m) for m in meta["max_y"]]
        est.scale_impute_ = blob["scale_impute"]
        est.network_ = TrunkMultiHead(
            len(est.feature_names_used_), est.targets_,
            num_neurons=est.num_neurons, act=est.act, out_act=est.out_act,
        ).double()
        est.network_.load_state_dict(blob["state_dict"])
        est.network_.eval()
        est.history_ = {}
        return est


class MODNetClassifier(BaseEstimator):
    """Categorical-head variant (softmax/sigmoid, cross-entropy). M4."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        raise NotImplementedError("milestone M4 — see DESIGN.md §2.3")

    def predict(self, X):
        raise NotImplementedError("milestone M4 — see DESIGN.md §2.3")
