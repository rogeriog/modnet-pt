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
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
)
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


def _train_network(network, x_all, y_list, head_weights, loss_fn, epochs,
                   batch_size, lr, rng, val_idx, train_idx,
                   patience: int = 0, min_delta: float = 0.0):
    """Shared Adam training loop with weighted per-head loss and optional
    early stopping. With a validation split, monitors ``val_loss``
    (sklearn-convention analog of upstream's EarlyStopping on validation);
    without one but with ``patience > 0``, monitors the training loss —
    exactly upstream's ``EarlyStopping(monitor="loss", min_delta=0.001,
    patience=100)`` used for every preset/GA candidate (vanilla.py:542-551).
    ``min_delta`` requires the monitored value to improve by at least that
    much to reset the patience counter (Keras semantics). Best weights are
    restored when early stopping triggers."""
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    history = {"loss": []}
    if len(val_idx):
        history["val_loss"] = []

    def _total_loss(preds, targets):
        total = torch.tensor(0.0, dtype=torch.float64)
        for pred, target, w in zip(preds, targets, head_weights):
            total = total + w * loss_fn(pred, target)
        return total

    best_val = float("inf")
    best_epoch = 0
    best_state = None
    network.train()
    for epoch in range(epochs):
        perm = rng.permutation(len(train_idx))
        losses = []
        for start in range(0, len(perm), batch_size):
            batch = train_idx[perm[start : start + batch_size]]
            xb = x_all[torch.tensor(batch)]
            yb = [y_i[torch.tensor(batch)] for y_i in y_list]
            optimizer.zero_grad()
            total = _total_loss(network(xb), yb)
            total.backward()
            optimizer.step()
            losses.append(total.detach().item())
        history["loss"].append(float(np.mean(losses)))
        if len(val_idx):
            network.eval()
            with torch.no_grad():
                total = _total_loss(
                    network(x_all[torch.tensor(val_idx)]),
                    [y_i[torch.tensor(val_idx)] for y_i in y_list],
                )
            val_loss = total.detach().item()
            history["val_loss"].append(val_loss)
            network.train()
            monitor_value = val_loss
        elif patience:
            monitor_value = history["loss"][-1]
        else:
            monitor_value = None
        if patience and monitor_value is not None:
            if monitor_value < best_val - min_delta:
                best_val = monitor_value
                best_epoch = epoch
                best_state = {
                    k: v.detach().clone() for k, v in network.state_dict().items()
                }
            elif epoch - best_epoch >= patience:
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    return history


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
                 search_folds: int = 5, presets=None,
                 size_pop: int = 20, num_generations: int = 10,
                 prob_mut=None, nested: int = 5, early_stopping: int = 4,
                 refit: int = 5,
                 bootstrap: int = 0, uncertainty: bool = False,
                 dropout=None,
                 val_fraction: float = 0.0, patience: int = 0,
                 min_delta: float = 0.0,
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
        self.search_folds = search_folds
        self.presets = presets
        self.size_pop = size_pop
        self.num_generations = num_generations
        self.prob_mut = prob_mut
        self.nested = nested
        self.early_stopping = early_stopping
        self.refit = refit
        self.bootstrap = bootstrap
        self.uncertainty = uncertainty
        self.dropout = dropout
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.device = device

    def _effective_dropout(self):
        """MC-dropout rate: explicit ``dropout`` wins; ``uncertainty=True``
        is the convenience alias for dropout=0.1 (DESIGN.md §2.3)."""
        if self.dropout is not None:
            return float(self.dropout)
        return 0.1 if self.uncertainty else 0.0

    def _check_config(self):
        if self.loss not in _LOSSES:
            raise ValueError(f"loss must be one of {_LOSSES}, got {self.loss!r}")
        if self.search not in ("none", "preset", "ga"):
            raise ValueError(f"unknown search mode {self.search!r}")
        if self.bootstrap < 0:
            raise ValueError("bootstrap must be >= 0")
        if self.refit < 0:
            raise ValueError("refit must be >= 0")

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

        if self.search == "preset":
            from modnet_pt.search import run_preset_search

            fitted, info = run_preset_search(
                self, X, y_frame, classification=False,
                presets=self.presets, n_splits=self.search_folds,
            )
            for key, value in fitted.__dict__.items():
                if key.endswith("_") and not key.startswith("__"):
                    setattr(self, key, value)
            self.preset_search_ = info
            self.n_features_in_ = X.shape[1]
            return self

        if self.search == "ga":
            if self.bootstrap:
                raise ValueError(
                    "bootstrap>0 is not supported with search='ga' — the GA "
                    "refit already produces an ensemble (refit param)"
                )
            from modnet_pt.search import run_ga_search

            members, _, info = run_ga_search(
                self, X, y_frame, classification=False,
                size_pop=self.size_pop, num_generations=self.num_generations,
                prob_mut=self.prob_mut, nested=self.nested,
                early_stopping=self.early_stopping, refit=self.refit,
                loss=self.loss, seed=self.random_state,
            )
            self.ensemble_ = members
            self.ga_results_ = info
            first = members[0]
            for key in (
                "targets_", "target_names_", "feature_names_used_",
                "min_y_", "max_y_", "scale_impute_", "network_",
                "history_", "n_features_in_", "feature_names_in_",
            ):
                setattr(self, key, getattr(first, key, None))
            self.n_features_in_ = X.shape[1]
            return self

        if self.bootstrap and self.search != "none":
            raise ValueError("bootstrap>0 requires search='none'")

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
            dropout=self._effective_dropout(),
        ).double()
        if self.loss == "mse":
            criterion = torch.nn.functional.mse_loss
        else:
            criterion = torch.nn.functional.l1_loss

        x_all = torch.tensor(x, dtype=torch.float64)
        self.history_ = _train_network(
            self.network_, x_all, y_list, head_weights, criterion,
            self.epochs, self.batch_size, self.lr, rng, val_idx, train_idx,
            patience=self.patience, min_delta=self.min_delta,
        )

        # Postprocessing bounds (vanilla.py:444-449)
        self.min_y_ = [y_frame[sub].values.min(axis=0) for sub in groups]
        self.max_y_ = [y_frame[sub].values.max(axis=0) for sub in groups]

        self.n_features_in_ = X.shape[1]

        if self.bootstrap:
            # EnsembleMODNetModel port (ensemble.py:79-117): n_models fits
            # on bootstrap resamples; predict returns the member mean.
            from modnet_pt.ensembles import fit_bootstrap_members

            template = clone(self)
            template.set_params(bootstrap=0, search="none")
            self.ensemble_ = fit_bootstrap_members(
                template, X, y_frame, self.bootstrap, self.random_state
            )
        return self

    def _scaled_X(self, X) -> np.ndarray:
        """Shared predict-time preprocessing: column selection, NaN
        guard, scale/impute (mirrors predict/_forward_probabilities)."""
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
        return self.scale_impute_.transform(x)

    def predict(self, X, remap_out_of_bounds: bool = True):
        """Predict target values for X.

        Returns a DataFrame with one column per target (joint-learning
        subgroups expand to their member columns), mirroring upstream
        ``MODNetModel.predict`` (vanilla.py:696-760) including the
        out-of-bounds remapping to the training range. For bootstrap/GA
        ensemble fits, returns the member mean (ensemble.py:146-187).
        """
        if getattr(self, "ensemble_", None):
            from modnet_pt.ensembles import ensemble_predict

            return ensemble_predict(self.ensemble_, X)
        check_is_fitted(self, "network_")
        x = self._scaled_X(X)

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

    def predict_std(self, X):
        """Epistemic uncertainty: std across bootstrap/GA ensemble members
        (ensemble.py:178-182). Only defined for ensemble fits."""
        if not getattr(self, "ensemble_", None):
            raise AttributeError(
                "predict_std requires an ensemble fit (bootstrap>0 or "
                "search='ga'); use predict_uncertainty for MC-dropout std"
            )
        from modnet_pt.ensembles import ensemble_predict_std

        return ensemble_predict_std(self.ensemble_, X)

    def predict_uncertainty(self, X, n_passes: int = 32):
        """MC-dropout epistemic uncertainty (DESIGN.md §2.3; analog of
        upstream ``bayesian.py:290``'s stochastic prediction loop, but with
        dropout instead of the deprecated DenseVariational layers).

        Runs ``n_passes`` stochastic forward passes with dropout active and
        BatchNorm frozen in eval mode, and returns ``(mean, std)``
        DataFrames. Requires ``dropout > 0`` (or ``uncertainty=True``) at
        fit time for a non-degenerate std; with dropout=0 all passes are
        identical and std is 0. For ensemble fits this reflects
        ``self.network_`` (the full-data model / first GA member).
        """
        check_is_fitted(self, "network_")
        x = self._scaled_X(X)
        columns = self.target_names_

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.network_.train()
        for module in self.network_.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
        passes = []
        try:
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float64)
                for _ in range(n_passes):
                    passes.append(
                        [p.numpy() for p in self.network_(xt)]
                    )
        finally:
            self.network_.eval()

        arr = np.stack(
            [np.hstack([p.reshape(len(p), -1) for p in ps]) for ps in passes]
        )  # (n_passes, n_samples, n_outputs)
        mean = pd.DataFrame(arr.mean(axis=0), columns=columns)
        std = pd.DataFrame(arr.std(axis=0), columns=columns)
        return mean, std

    def _member_blob(self, member):
        return {
            "state_dict": member.network_.state_dict(),
            "meta_json": json.dumps(
                {
                    "params": member.get_params(),
                    "targets": member.targets_,
                    "target_names": member.target_names_,
                    "feature_names_used": member.feature_names_used_,
                    "min_y": [m.tolist() for m in member.min_y_],
                    "max_y": [m.tolist() for m in member.max_y_],
                }
            ),
            "scale_impute": member.scale_impute_,
        }

    def save(self, path: str):
        """Save fitted model: torch state_dict + metadata JSON (+ ensemble
        members when fitted with ``bootstrap``/``search="ga"``).

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
        blob = {
            "state_dict": self.network_.state_dict(),
            "meta_json": json.dumps(meta),
            "scale_impute": self.scale_impute_,
        }
        if getattr(self, "ensemble_", None):
            blob["ensemble"] = [self._member_blob(m) for m in self.ensemble_]
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str) -> MODNetRegressor:
        """Restore a model saved with ``save``."""
        blob = torch.load(path, weights_only=False)
        est = cls._from_blob(blob)
        if blob.get("ensemble"):
            est.ensemble_ = [
                cls._from_blob(member_blob) for member_blob in blob["ensemble"]
            ]
        return est

    @classmethod
    def _from_blob(cls, blob) -> MODNetRegressor:
        """Rebuild one estimator from a save() blob (used for ensemble
        members)."""
        meta = json.loads(blob["meta_json"])
        params = meta["params"]
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
            dropout=est._effective_dropout(),
        ).double()
        est.network_.load_state_dict(blob["state_dict"])
        est.network_.eval()
        est.history_ = {}
        return est

    def _preset_val_score(self, X, y, loss: str = "mae"):
        """Validation score for preset/GA candidate selection: mean per-
        subgroup loss on raw predictions (upstream ``evaluate``,
        vanilla.py:768-839 — MAE by default, the genome's loss for GA)."""
        from modnet_pt.search import regression_preset_score

        y_frame, _ = self._prepare_y(y)
        pred = self.predict(X, remap_out_of_bounds=False)
        return regression_preset_score(self.targets_, y_frame, pred, loss=loss)


class MODNetClassifier(ClassifierMixin, BaseEstimator):
    """MODNet classifier: softmax (multi-class) or sigmoid (multi-label)
    heads with cross-entropy loss.

    Port of the upstream classification path (vanilla.py:186-199 heads,
    :344-362 one-hot targets, categorical/binary cross-entropy): ``y`` is a
    single class column (Series / 1-D array / one-column DataFrame) or a
    DataFrame with one class column per target (joint classification, one
    head per column). ``multi_label=True`` switches heads to per-class
    sigmoid + binary cross-entropy (upstream ``multi_label``).

    ``predict`` returns a DataFrame with the class labels; ``predict_proba``
    returns a DataFrame whose columns are the class values.
    """

    def __init__(self, targets=None, weights=None, n_feat: int = 64,
                 num_neurons=([64], [32], [16], [16]), act: str = "relu",
                 lr: float = 1e-3, epochs: int = 200,
                 batch_size: int = 128, xscale: str = "minmax",
                 impute_missing=0, multi_label: bool = False,
                 search: str = "none", search_folds: int = 5, presets=None,
                 size_pop: int = 20, num_generations: int = 10,
                 prob_mut=None, nested: int = 5, early_stopping: int = 4,
                 refit: int = 5, dropout=None,
                 val_fraction: float = 0.0, patience: int = 0,
                 min_delta: float = 0.0,
                 random_state: int | None = None, device: str = "cpu"):
        self.targets = targets
        self.weights = weights
        self.n_feat = n_feat
        self.num_neurons = num_neurons
        self.act = act
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.xscale = xscale
        self.impute_missing = impute_missing
        self.multi_label = multi_label
        self.search = search
        self.search_folds = search_folds
        self.presets = presets
        self.size_pop = size_pop
        self.num_generations = num_generations
        self.prob_mut = prob_mut
        self.nested = nested
        self.early_stopping = early_stopping
        self.refit = refit
        self.dropout = dropout
        self.val_fraction = val_fraction
        self.patience = patience
        self.min_delta = min_delta
        self.random_state = random_state
        self.device = device

    def _effective_dropout(self):
        return 0.0 if self.dropout is None else float(self.dropout)

    def _prepare_y(self, y):
        if isinstance(y, pd.Series):
            y = y.to_frame()
        if isinstance(y, pd.DataFrame):
            return y
        arr = np.asarray(y)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        names = [f"y{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=names)

    def fit(self, X, y):
        if self.search not in ("none", "preset", "ga"):
            raise ValueError(f"unknown search mode {self.search!r}")

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        else:
            X = pd.DataFrame(np.asarray(X))

        y_frame = self._prepare_y(y)
        if len(X) != len(y_frame):
            raise ValueError("X and y have different numbers of samples.")
        if y_frame.isna().any().any():
            raise ValueError("y must not contain missing class labels.")

        if self.search == "preset":
            from modnet_pt.search import run_preset_search

            fitted, info = run_preset_search(
                self, X, y_frame, classification=True,
                presets=self.presets, n_splits=self.search_folds,
            )
            for key, value in fitted.__dict__.items():
                if key.endswith("_") and not key.startswith("__"):
                    setattr(self, key, value)
            self.preset_search_ = info
            self.n_features_in_ = X.shape[1]
            return self

        if self.search == "ga":
            from modnet_pt.search import run_ga_search

            members, _, info = run_ga_search(
                self, X, y_frame, classification=True,
                size_pop=self.size_pop, num_generations=self.num_generations,
                prob_mut=self.prob_mut, nested=self.nested,
                early_stopping=self.early_stopping, refit=self.refit,
                loss="categorical_crossentropy", seed=self.random_state,
            )
            self.ensemble_ = members
            self.ga_results_ = info
            first = members[0]
            for key in (
                "targets_", "target_names_", "feature_names_used_",
                "classes_", "scale_impute_", "network_", "history_",
                "n_features_in_", "feature_names_in_",
            ):
                setattr(self, key, getattr(first, key, None))
            self.n_features_in_ = X.shape[1]
            return self

        targets = _normalize_targets(self.targets, list(y_frame.columns))
        # classification heads are one-per-column: flatten subgroups to
        # single-target subgroups (upstream n >= 2 heads, vanilla.py:186)
        targets = [[[prop] for group in targets for sub in group for prop in sub]]
        self.targets_ = targets
        columns = [p for group in targets for sub in group for p in sub]
        missing = [c for c in columns if c not in y_frame.columns]
        if missing:
            raise ValueError(f"targets not present in y: {missing}")

        if self.n_feat > X.shape[1]:
            raise ValueError(
                "The model requires more features than computed in data. "
                f"Please reduce n_feat below or equal to {X.shape[1]}"
            )
        self.feature_names_used_ = list(X.columns)[: self.n_feat]
        x = X[self.feature_names_used_].values.astype(np.float64)

        # label encoding per target column (upstream to_categorical,
        # vanilla.py:347-353; kept as class index tensors here).
        # multi_label: one binary sigmoid head per column (width 1).
        self.classes_ = {}
        num_classes = {}
        y_encoded = []
        for col in columns:
            labels = y_frame[col].values
            classes = np.unique(labels)
            if len(classes) < 2:
                raise ValueError(
                    f"target {col!r} needs at least 2 classes, got {len(classes)}"
                )
            if self.multi_label and len(classes) != 2:
                raise ValueError(
                    f"multi_label target {col!r} needs exactly 2 classes, "
                    f"got {len(classes)}"
                )
            self.classes_[col] = classes
            if self.multi_label:
                num_classes[col] = 0  # regression-width head → single logit
                y_encoded.append(
                    torch.tensor(
                        (labels == classes[1]).astype(np.float64)
                    ).unsqueeze(1)
                )
            else:
                num_classes[col] = len(classes)
                y_encoded.append(
                    torch.tensor(
                        np.searchsorted(classes, labels).astype(np.int64),
                        dtype=torch.float64,
                    ).unsqueeze(1)
                )

        weights = self.weights or {}
        head_weights = [float(weights.get(col, 1)) for col in columns]

        self.scale_impute_ = ScaleImpute(
            xscale=self.xscale, impute_missing=self.impute_missing
        ).fit(x)
        x = self.scale_impute_.transform(x)

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
            num_classes=num_classes, act=self.act,
            dropout=self._effective_dropout(),
        ).double()

        if self.multi_label:
            def loss_fn(pred, target):
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    pred, target
                )
        else:
            def loss_fn(pred, target):
                return torch.nn.functional.cross_entropy(
                    pred, target.squeeze(1).long()
                )

        x_all = torch.tensor(x, dtype=torch.float64)
        self.history_ = _train_network(
            self.network_, x_all, y_encoded, head_weights, loss_fn,
            self.epochs, self.batch_size, self.lr, rng, val_idx, train_idx,
            patience=self.patience, min_delta=self.min_delta,
        )
        self.target_names_ = columns
        self.n_features_in_ = X.shape[1]
        return self

    def _preset_val_score(self, X, y, loss: str = "mae"):
        """Validation score for preset/GA candidate selection: mean negative
        one-vs-rest ROC AUC (upstream ``evaluate`` classification branch,
        vanilla.py:802-821 — ``loss`` is ignored for classification, as
        upstream)."""
        from modnet_pt.search import classification_preset_score

        y_frame = self._prepare_y(y)
        proba = self.predict_proba(X)
        return classification_preset_score(
            self.classes_, self.target_names_, self.multi_label, proba, y_frame
        )

    def _scaled_X(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names_used_ if c not in X.columns]
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
        return self.scale_impute_.transform(x)

    def _forward_probabilities(self, X):
        check_is_fitted(self, "network_")
        x = self._scaled_X(X)

        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(torch.tensor(x, dtype=torch.float64))
        if self.multi_label:
            return [torch.sigmoid(out).numpy() for out in logits]
        return [torch.softmax(out, dim=-1).numpy() for out in logits]

    def _proba_frame(self, probs):
        columns = self.target_names_
        tuples = []
        for col, prob in zip(columns, probs):
            if self.multi_label:
                tuples.append((col, self.classes_[col][1]))
            else:
                tuples.extend((col, cls) for cls in self.classes_[col])
        out = np.hstack([prob.reshape(len(prob), -1) for prob in probs])
        index = pd.MultiIndex.from_tuples(tuples, names=["target", "class"])
        return pd.DataFrame(out, columns=index)

    def predict_proba(self, X):
        """Class probabilities: DataFrame with one column per class,
        MultiIndexed by (target, class). For multi-label fits, one column
        per target holding P(positive class). Ensemble fits return the
        member-mean probabilities."""
        if getattr(self, "ensemble_", None):
            from modnet_pt.ensembles import ensemble_predict_proba

            return ensemble_predict_proba(self.ensemble_, X)
        return self._proba_frame(self._forward_probabilities(X))

    def predict(self, X):
        """Predicted class labels per target column (argmax; multi-label
        fits threshold the positive-class probability at 0.5). Ensemble
        fits argmax the member-mean probabilities."""
        if getattr(self, "ensemble_", None):
            proba = self.predict_proba(X)
            preds = {}
            for col in self.target_names_:
                sub = proba[col]
                if self.multi_label:
                    positive = sub.iloc[:, 0].to_numpy() > 0.5
                    preds[col] = self.classes_[col][positive.astype(int)]
                else:
                    preds[col] = self.classes_[col][
                        np.argmax(sub.to_numpy(), axis=1)
                    ]
            return pd.DataFrame(preds)
        probs = self._forward_probabilities(X)
        columns = self.target_names_
        preds = {}
        for col, prob in zip(columns, probs):
            if self.multi_label:
                positive = prob[:, 0] > 0.5
                preds[col] = self.classes_[col][positive.astype(int)]
            else:
                preds[col] = self.classes_[col][np.argmax(prob, axis=1)]
        return pd.DataFrame(preds)

    def predict_std(self, X):
        """Std of member class-probabilities across an ensemble fit
        (GA ``refit``); only defined for ensemble fits."""
        if not getattr(self, "ensemble_", None):
            raise AttributeError(
                "predict_std requires an ensemble fit (search='ga')"
            )
        from modnet_pt.ensembles import ensemble_predict_proba_std

        return ensemble_predict_proba_std(self.ensemble_, X)

    def predict_uncertainty(self, X, n_passes: int = 32):
        """MC-dropout uncertainty: ``n_passes`` stochastic forward passes
        (dropout active, BatchNorm frozen) -> ``(mean_probs, std_probs)``
        MultiIndexed DataFrames. See ``MODNetRegressor.predict_uncertainty``.
        """
        check_is_fitted(self, "network_")
        x = self._scaled_X(X)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        self.network_.train()
        for module in self.network_.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.eval()
        passes = []
        try:
            with torch.no_grad():
                xt = torch.tensor(x, dtype=torch.float64)
                for _ in range(n_passes):
                    logits = self.network_(xt)
                    if self.multi_label:
                        probs = [torch.sigmoid(o) for o in logits]
                    else:
                        probs = [torch.softmax(o, dim=-1) for o in logits]
                    passes.append([p.numpy() for p in probs])
        finally:
            self.network_.eval()

        arr = np.stack(
            [np.hstack([p.reshape(len(p), -1) for p in ps]) for ps in passes]
        )
        # rebuild the (target, class) column index once, reuse for both
        index = self._proba_frame(passes[0]).columns
        return (
            pd.DataFrame(arr.mean(axis=0), columns=index),
            pd.DataFrame(arr.std(axis=0), columns=index),
        )

    def _member_blob(self, member):
        return {
            "state_dict": member.network_.state_dict(),
            "meta_json": json.dumps(
                {
                    "params": member.get_params(),
                    "targets": member.targets_,
                    "target_names": member.target_names_,
                    "feature_names_used": member.feature_names_used_,
                    "classes": {
                        k: v.tolist() for k, v in member.classes_.items()
                    },
                }
            ),
            "scale_impute": member.scale_impute_,
        }

    def save(self, path: str):
        """Save fitted classifier: torch state_dict + metadata JSON
        (+ ensemble members for GA fits), mirroring
        ``MODNetRegressor.save`` (DESIGN.md §2.3)."""
        check_is_fitted(self, "network_")
        meta = {
            "params": self.get_params(),
            "targets": self.targets_,
            "target_names": self.target_names_,
            "feature_names_used": self.feature_names_used_,
            "classes": {k: v.tolist() for k, v in self.classes_.items()},
        }
        blob = {
            "state_dict": self.network_.state_dict(),
            "meta_json": json.dumps(meta),
            "scale_impute": self.scale_impute_,
        }
        if getattr(self, "ensemble_", None):
            blob["ensemble"] = [self._member_blob(m) for m in self.ensemble_]
        torch.save(blob, path)

    @classmethod
    def _from_blob(cls, blob) -> MODNetClassifier:
        meta = json.loads(blob["meta_json"])
        params = meta["params"]
        params["num_neurons"] = tuple(
            list(block) for block in params["num_neurons"]
        )
        est = cls(**params)
        est.targets_ = [[list(sub) for sub in group] for group in meta["targets"]]
        est.target_names_ = meta["target_names"]
        est.feature_names_used_ = meta["feature_names_used"]
        est.classes_ = {k: np.array(v) for k, v in meta["classes"].items()}
        est.scale_impute_ = blob["scale_impute"]
        num_classes = {
            col: (0 if est.multi_label else len(est.classes_[col]))
            for col in est.target_names_
        }
        est.network_ = TrunkMultiHead(
            len(est.feature_names_used_), est.targets_,
            num_neurons=est.num_neurons, num_classes=num_classes,
            act=est.act, dropout=est._effective_dropout(),
        ).double()
        est.network_.load_state_dict(blob["state_dict"])
        est.network_.eval()
        est.history_ = {}
        return est

    @classmethod
    def load(cls, path: str) -> MODNetClassifier:
        """Restore a classifier saved with ``save``."""
        blob = torch.load(path, weights_only=False)
        est = cls._from_blob(blob)
        if blob.get("ensemble"):
            est.ensemble_ = [
                cls._from_blob(member_blob) for member_blob in blob["ensemble"]
            ]
        return est
