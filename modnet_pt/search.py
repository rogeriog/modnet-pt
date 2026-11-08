"""Hyperparameter search: preset grid + genetic algorithm.

Port targets (upstream v0.4.4 @ 1281977):
  modnet/model_presets/presets.py:12   gen_presets grid
  modnet/models/vanilla.py:478         fit_preset inner-validation loop
  modnet/models/vanilla.py:1477        validate_model (per-candidate fit + score)
  modnet/matbench/benchmark.py:16      matbench_kfold_splits (fixed-seed KFold)
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

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

# upstream benchmark.py:13 — fixed seed so preset/GA splits are reproducible
MATBENCH_SEED = 18012019


def gen_presets(n_feat: int, n_samples: int, classification: bool = False):
    """Preset hyperparameter grid (presets.py:12-81).

    Same rules: batch sizes by sample count, lr grid, single epoch count,
    loss by task type, elu activation, minmax/standard scaling, and three
    architectures per filtered feature count. Deviations (documented):
    empty ``n_feat_list`` falls back to ``[n_feat]`` and widths are clamped
    to >= 1 — upstream produces an empty grid / zero-width layers for tiny
    feature counts.
    """
    if n_samples < 1000:
        batch_sizes = [32, 64]
    else:
        batch_sizes = [64]
    learning_rates = [0.001, 0.005, 0.01]
    epochs = [1000]
    losses = ["categorical_crossentropy"] if classification else ["mae"]
    activations = ["elu"]
    xscale = ["minmax", "standard"]

    n_feat_list = [n for n in (64, 128, 256, 512) if n_feat / 20 < n <= n_feat]
    if not n_feat_list:
        n_feat_list = [n_feat]
    if len(n_feat_list) == 1:
        n_feat_list.append(n_feat)
    if len(n_feat_list) < 3:
        n_feat_list.append((n_feat_list[0] + n_feat_list[1]) // 2)
    n_feat_list = sorted(set(n_feat_list))

    def _w(nf, k):
        return max(1, nf // k)

    archs = []
    for nf in n_feat_list:
        archs += [
            (nf, [[nf * 2], [_w(nf, 2)], [_w(nf, 8)], [_w(nf, 8)]]),
            (nf, [[nf], [_w(nf, 2)], [_w(nf, 8)], [_w(nf, 8)]]),
            (nf, [[_w(nf, 2)], [_w(nf, 4)], [_w(nf, 8)], [_w(nf, 8)]]),
        ]

    import itertools

    presets = []
    for a, bs, lr, e, l, act, scaler in itertools.product(
        archs, batch_sizes, learning_rates, epochs, losses, activations, xscale
    ):
        presets.append(
            {
                "batch_size": bs,
                "lr": lr,
                "n_feat": a[0],
                "num_neurons": a[1],
                "epochs": e,
                "loss": l,
                "act": act,
                "xscale": scaler,
            }
        )
    return presets


def kfold_splits(n_samples: int, y_first=None, n_splits: int = 5,
                 classification: bool = False, seed: int = MATBENCH_SEED):
    """Inner-CV splits, port of ``matbench_kfold_splits``
    (benchmark.py:16-42): shuffled KFold for regression, StratifiedKFold on
    the first target column for classification, fixed ``MATBENCH_SEED``.
    ``n_splits <= 1`` is coerced to 5 (upstream fit_preset, vanilla.py:571)."""
    if n_splits <= 1:
        n_splits = 5
    if classification:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=seed)
        return list(kf.split(np.zeros(n_samples), y_first))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(np.zeros(n_samples)))


def _candidate_params(estimator, preset, n_feat_avail):
    """Map a preset dict onto estimator params (intersection with
    get_params — e.g. classifiers have no ``loss`` param) plus the
    fixed early-stopping rule upstream applies to every candidate
    (vanilla.py:542-551: monitor train loss, patience 100, min_delta 1e-3)."""
    keys = set(estimator.get_params())
    params = {k: v for k, v in preset.items() if k in keys}
    if "n_feat" in params:
        params["n_feat"] = min(n_feat_avail, params["n_feat"])
    params.update(search="none", patience=100, min_delta=0.001)
    return params


def run_preset_search(estimator, X: pd.DataFrame, y: pd.DataFrame,
                      *, classification: bool, presets=None, n_splits: int = 5):
    """Preset grid search with inner k-fold validation, then refit on the
    full training set — the sklearn-ified ``fit_preset`` (vanilla.py:478).

    ``estimator`` must be unfitted with ``search="preset"`` (or ``"ga"``
    once M6 lands); it is used as the clone template. Returns
    ``(fitted_estimator, info)`` where ``info`` carries the preset list,
    the (n_presets × n_splits) score matrix and the winning preset.

    Deviations from upstream, all documented: candidates early-stop on the
    training loss like upstream's EarlyStopping(monitor="loss") but restore
    the best weights (upstream ``restore_best_weights=False``); the refit
    uses the winning preset's ``xscale`` — upstream ``fit_preset`` forgets
    to pass it and silently refits with the default ``"minmax"``
    (vanilla.py:674-684).
    """
    n_feat_avail = X.shape[1]
    if presets is None:
        presets = gen_presets(n_feat_avail, len(X),
                              classification=classification)
    if not presets:
        raise ValueError("preset search requires at least one preset")

    y_first = y.iloc[:, 0].values if classification else None
    splits = kfold_splits(len(X), y_first=y_first, n_splits=n_splits,
                          classification=classification)

    scores = np.full((len(presets), len(splits)), np.inf)
    base_seed = getattr(estimator, "random_state", None)
    for i, preset in enumerate(presets):
        for j, (train_idx, val_idx) in enumerate(splits):
            cand = clone(estimator)
            cand.set_params(**_candidate_params(cand, preset, n_feat_avail))
            if base_seed is not None:
                # upstream TF inits every fit differently; derive a
                # reproducible per-candidate seed (deterministic given
                # random_state, diverse across preset/fold like upstream)
                cand.set_params(random_state=base_seed * 100003 + i * 131 + j)
            cand.fit(X.iloc[train_idx], y.iloc[train_idx])
            scores[i, j] = cand._preset_val_score(X.iloc[val_idx],
                                                  y.iloc[val_idx])

    mean_scores = scores.mean(axis=1)
    best_idx = int(np.nanargmin(mean_scores))
    best = presets[best_idx]

    final = clone(estimator)
    final.set_params(**_candidate_params(final, best, n_feat_avail))
    if base_seed is not None:
        final.set_params(random_state=base_seed * 100003 + 999983)
    final.fit(X, y)

    info = {
        "presets": presets,
        "scores": scores,
        "mean_scores": mean_scores,
        "best_preset_idx": best_idx,
        "best_preset": best,
        "n_splits": len(splits),
    }
    return final, info


def regression_preset_score(targets_, y_true: pd.DataFrame,
                            pred: pd.DataFrame, loss: str = "mae") -> float:
    """Mean per-subgroup loss — upstream ``evaluate`` regression branch
    (vanilla.py:822-839) on raw (un-remapped) predictions. ``loss`` is the
    candidate's genome loss for GA (``"mae"``/``"mse"``)."""
    metric = mean_absolute_error if loss == "mae" else mean_squared_error
    scores = [
        metric(y_true[sub].values, pred[sub].values)
        for group in targets_
        for sub in group
    ]
    return float(np.mean(scores))


def classification_preset_score(classes_, target_names_, multi_label: bool,
                                proba: pd.DataFrame,
                                y_true: pd.DataFrame) -> float:
    """Mean negative one-vs-rest ROC AUC over target columns — upstream
    ``evaluate`` classification branch (vanilla.py:802-821). Per-class AUCs
    are averaged within a target; a class absent from the fold contributes
    NaN and is dropped (upstream try/except ValueError → nanmean)."""
    per_target = []
    for col in target_names_:
        classes = classes_[col]
        labels = y_true[col].values
        if multi_label:
            positive = classes[1]
            p = proba[(col, positive)].to_numpy().ravel()
            yt = (labels == positive).astype(int)
            try:
                auc = roc_auc_score(yt, p)
            except ValueError:
                auc = np.nan
            per_target.append(-auc)
        else:
            aucs = []
            for cls in classes:
                yt = (labels == cls).astype(int)
                p = proba[(col, cls)].to_numpy().ravel()
                try:
                    aucs.append(roc_auc_score(yt, p))
                except ValueError:
                    aucs.append(np.nan)
            per_target.append(-float(np.nanmean(aucs)))
    score = float(np.mean(per_target))
    return score if np.isfinite(score) else np.inf


# ---------------------------------------------------------------------------
# Genetic algorithm (milestone M6) — port of modnet/hyper_opt/fit_genetic.py
# ---------------------------------------------------------------------------

_FRACTION_LIST = [1, 0.75, 0.5, 0.25]
_LR_LIST = [0.1, 0.01, 0.005, 0.001]
_BATCH_LIST = [32, 64, 128, 256]
_XSCALE_LIST = ["minmax", "standard"]
_IMPUTE_LIST = [0, "mean"]
_GA_EPOCHS = 800  # fit_genetic.py:227/:299


class Individual:
    """GA genome: a dict of hyperparameters plus crossover/mutation
    (fit_genetic.py:15-164). ``rng`` is a ``np.random.Generator`` — upstream
    uses the unseeded ``random``/``np.random`` globals, so runs are
    non-deterministic there; ours are seeded via the estimator's
    ``random_state`` (documented deviation)."""

    def __init__(self, max_feat: int, rng: np.random.Generator,
                 loss: str = "mae"):
        self.act = "elu"
        self.loss = loss
        self.n_neurons_first_layer = 32 * int(rng.integers(1, 11))
        self.max_feat = max_feat
        self.genes = {
            "act": self.act,
            "loss": self.loss,
            "n_neurons_first_layer": self.n_neurons_first_layer,
            "fraction1": _FRACTION_LIST[rng.integers(len(_FRACTION_LIST))],
            "fraction2": _FRACTION_LIST[rng.integers(len(_FRACTION_LIST))],
            "fraction3": _FRACTION_LIST[rng.integers(len(_FRACTION_LIST))],
            "xscale": _XSCALE_LIST[rng.integers(len(_XSCALE_LIST))],
            "impute_missing": _IMPUTE_LIST[rng.integers(len(_IMPUTE_LIST))],
            "lr": _LR_LIST[rng.integers(len(_LR_LIST))],
            "batch_size": _BATCH_LIST[rng.integers(len(_BATCH_LIST))],
            "n_feat": self._sample_n_feat(max_feat, rng),
        }

    @staticmethod
    def _sample_n_feat(max_feat: int, rng: np.random.Generator) -> int:
        """n_feat gene ranges (fit_genetic.py:75-83)."""
        max_feat = max(2, int(max_feat))
        if max_feat <= 100:
            b = int(max_feat / 2)
            return int(rng.integers(1, b + 1)) + b
        if max_feat < 2000:
            return 10 * int(rng.integers(1, int(max_feat / 10) + 1))
        m = int(np.sqrt(max_feat))
        return int(rng.integers(1, m + 1)) ** 2

    def crossover(self, partner: Individual,
                  rng: np.random.Generator) -> Individual:
        """Uniform crossover: half the genes from each parent
        (fit_genetic.py:85-110)."""
        keys = list(self.genes)
        mother = set(
            rng.choice(keys, size=len(keys) // 2, replace=False).tolist()
        )
        child_genes = {
            g: self.genes[g] if g in mother else partner.genes[g]
            for g in keys
        }
        child = Individual(self.max_feat, rng, loss=child_genes["loss"])
        child.genes = child_genes
        return child

    def mutation(self, prob_mut: float, rng: np.random.Generator) -> None:
        """In-place mutation (fit_genetic.py:112-164): n_feat ±10% of
        max_feat, first-layer width ±64 clamped away from 0, one fraction
        resampled, batch size halved/doubled. Deviations: the fraction pick
        is a proper uniform choice — upstream ``random.choices([1,2,3])``
        returns a list, so the ``== 1``/``== 2`` branches never fire and
        fraction3 is *always* the mutated one (upstream bug); widths are
        clamped to >= 1 downstream in ``to_params``."""
        if rng.random() >= prob_mut:
            return
        span = int(0.1 * self.max_feat)
        n_feat = int(
            self.genes["n_feat"]
            + rng.integers(-span, span + 1) if span else self.genes["n_feat"]
        )
        self.genes["n_feat"] = min(max(1, abs(n_feat)), self.max_feat)
        nfl = abs(
            int(self.genes["n_neurons_first_layer"])
            + 32 * int(rng.integers(-2, 3))
        )
        self.genes["n_neurons_first_layer"] = nfl if nfl else 32
        frac_key = f"fraction{int(rng.integers(1, 4))}"
        self.genes[frac_key] = _FRACTION_LIST[rng.integers(len(_FRACTION_LIST))]
        self.genes["batch_size"] = int(
            self.genes["batch_size"] * 2 ** int(rng.integers(-1, 2))
        )

    def num_neurons(self):
        """Derived architecture (fit_genetic.py:194-212), widths clamped
        to >= 1 (upstream can emit 0-unit layers)."""
        nfl = int(self.genes["n_neurons_first_layer"])
        f1 = self.genes["fraction1"]
        f2 = self.genes["fraction2"]
        f3 = self.genes["fraction3"]
        return [
            [nfl],
            [max(1, int(nfl * f1))],
            [max(1, int(nfl * f1 * f2))],
            [max(1, int(nfl * f1 * f2 * f3))],
        ]

    def to_params(self, n_feat_avail: int, estimator_params) -> dict:
        """Genome -> estimator params, intersected with the estimator's own
        ``get_params`` (classifiers have no ``loss``). Adds the GA
        early-stopping rule (fit_genetic.py:180-188/:250-258: monitor train
        loss, patience 30, min_delta 1e-3, 800 epochs)."""
        raw = {
            "n_feat": min(int(self.genes["n_feat"]), n_feat_avail),
            "num_neurons": self.num_neurons(),
            "act": self.genes["act"],
            "lr": self.genes["lr"],
            "batch_size": self.genes["batch_size"],
            "epochs": _GA_EPOCHS,
            "xscale": self.genes["xscale"],
            "impute_missing": self.genes["impute_missing"],
            "loss": self.genes["loss"],
            "search": "none",
            "patience": 30,
            "min_delta": 0.001,
        }
        return {k: v for k, v in raw.items() if k in estimator_params}


def _ga_fitness(estimator, individuals, X, y, splits, n_feat_avail,
                classification):
    """Mean nested-CV score per individual — ``function_fitness``
    (fit_genetic.py:413-508) without multiprocessing (serial port;
    documented deviation)."""
    scores = np.full(len(individuals), np.inf)
    keys = set(estimator.get_params())
    base_seed = getattr(estimator, "random_state", None)
    for i, ind in enumerate(individuals):
        fold_scores = []
        for j, (train_idx, val_idx) in enumerate(splits):
            cand = clone(estimator)
            params = {
                k: v for k, v in ind.to_params(n_feat_avail, keys).items()
            }
            cand.set_params(**params)
            if base_seed is not None:
                cand.set_params(random_state=base_seed * 100003 + i * 131 + j)
            cand.fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_scores.append(
                cand._preset_val_score(
                    X.iloc[val_idx], y.iloc[val_idx],
                    loss=ind.genes["loss"],
                )
            )
        scores[i] = float(np.mean(fold_scores))
    return scores


def run_ga_search(estimator, X: pd.DataFrame, y: pd.DataFrame, *,
                  classification: bool, size_pop: int = 20,
                  num_generations: int = 10, prob_mut=None, nested: int = 5,
                  early_stopping: int = 4, refit: int = 5, loss: str = "mae",
                  seed=None):
    """Genetic-algorithm hyperparameter search — ``FitGenetic.run``
    (fit_genetic.py:510-667) as an sklearn ``fit`` mode.

    Mirrors upstream: fitness = mean nested-CV score (same
    ``matbench_kfold_splits`` splits as preset search), parent selection
    weighted by ``1/loss**5`` over the *ranked* population, uniform
    crossover + mutation, elitist ranking over every individual ever
    evaluated, early stop after ``early_stopping`` generations without a
    new best, then refit: ``refit > 0`` fits a bootstrap ensemble of the
    winning genome on the full data (upstream ``refit_model`` →
    ``EnsembleMODNetModel``, fit_genetic.py:243-310); ``refit == 0``
    ensembles the fold models of the top-10 individuals.

    Deviations: serial evaluation (no multiprocessing), seeded RNG,
    fraction-mutation bug fixed, layer widths clamped >= 1, best weights
    restored on early stop.
    """
    from modnet_pt.ensembles import fit_bootstrap_members

    rng = np.random.default_rng(
        seed if seed is not None else (estimator.random_state or 0)
    )
    n_feat_avail = X.shape[1]
    n_splits = nested if nested and nested > 1 else 5
    y_first = y.iloc[:, 0].values if classification else None
    splits = kfold_splits(len(X), y_first=y_first, n_splits=n_splits,
                          classification=classification)

    population = [
        Individual(n_feat_avail, rng, loss=loss) for _ in range(size_pop)
    ]
    all_individuals = list(population)
    all_scores = list(_ga_fitness(estimator, population, X, y, splits,
                                  n_feat_avail, classification))

    best_per_gen = []
    ranking = np.argsort(all_scores)
    best_per_gen.append(all_individuals[ranking[0]])
    generations_run = 1

    for gen in range(1, num_generations):
        ranking = np.argsort(all_scores)
        ranked = [all_individuals[i] for i in ranking]
        # Parent weights 1/loss**5 over the ranked population
        # (fit_genetic.py:573-577). Scores are shifted positive first:
        # classification fitness is -AUC (negative), which would produce
        # negative weights — upstream's random.choices would crash there;
        # the shift preserves ranking and relative selection pressure.
        raw = np.asarray(all_scores, dtype=float)[ranking]
        finite = raw[np.isfinite(raw)]
        offset = finite.min() if len(finite) else 0.0
        shifted = raw - offset + 1e-6
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = 1.0 / shifted**5
        weights = np.where(np.isnan(weights), 1e-5, weights)
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
        if weights.sum() <= 0:
            weights = np.full(len(ranked), 1.0 / len(ranked))
        else:
            weights = weights / weights.sum()

        parents1 = rng.choice(len(ranked), size=size_pop, replace=True,
                              p=weights)
        parents2 = rng.choice(len(ranked), size=size_pop, replace=True,
                              p=weights)
        children = [
            ranked[parents1[i]].crossover(ranked[parents2[i]], rng)
            for i in range(size_pop)
        ]
        pm = prob_mut if prob_mut is not None else 1.0 / size_pop
        for child in children:
            child.mutation(pm, rng)

        child_scores = _ga_fitness(estimator, children, X, y, splits,
                                   n_feat_avail, classification)
        all_individuals.extend(children)
        all_scores.extend(child_scores)
        generations_run = gen + 1

        ranking = np.argsort(all_scores)
        best_per_gen.append(all_individuals[ranking[0]])
        if (
            gen >= early_stopping - 1
            and all(
                best_per_gen[k] is best_per_gen[gen]
                for k in range(gen - early_stopping + 1, gen + 1)
            )
        ):
            break

    ranking = np.argsort(all_scores)
    best_individual = all_individuals[ranking[0]]
    best_params = best_individual.to_params(n_feat_avail,
                                            set(estimator.get_params()))

    if refit:
        # upstream refit_model: EnsembleMODNetModel(n_models=refit) of
        # bootstrap fits with the winning genome (fit_genetic.py:655-657)
        template = clone(estimator)
        template.set_params(**best_params)
        members = fit_bootstrap_members(
            template, X, y, n_models=refit,
            seed=int(rng.integers(0, 2**31)),
        )
    else:
        # upstream refit=0: ensemble the fold models of the top-10
        # individuals (fit_genetic.py:659-663)
        members = []
        for ind in [all_individuals[i] for i in ranking[:10]]:
            params = ind.to_params(n_feat_avail, set(estimator.get_params()))
            for train_idx, _ in splits:
                member = clone(estimator)
                member.set_params(**params)
                member.fit(X.iloc[train_idx], y.iloc[train_idx])
                members.append(member)

    info = {
        "genes": dict(best_individual.genes),
        "scores": np.asarray(all_scores),
        "generations_run": generations_run,
        "n_evaluated": len(all_scores),
        "refit": refit,
    }
    return members, best_params, info
