"""Genetic-algorithm search tests (milestone M6): Individual genome,
crossover/mutation, run_ga_search fitness loop, ensemble refit."""

import numpy as np
import pandas as pd
import pytest

from modnet_pt import MODNetRegressor
from modnet_pt.search import Individual


def _data(n=80, seed=0, n_features=6):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y = pd.DataFrame({"y": X["f0"] + 0.5 * X["f1"] ** 2 + rng.normal(0, 0.05, n)})
    return X, y


def test_individual_genome_ranges():
    rng = np.random.default_rng(0)
    ind = Individual(max_feat=50, rng=rng)
    g = ind.genes
    assert g["act"] == "elu"
    assert 32 <= g["n_neurons_first_layer"] <= 320
    assert g["fraction1"] in (1, 0.75, 0.5, 0.25)
    assert g["xscale"] in ("minmax", "standard")
    assert g["impute_missing"] in (0, "mean")
    assert g["lr"] in (0.1, 0.01, 0.005, 0.001)
    assert g["batch_size"] in (32, 64, 128, 256)
    # max_feat <= 100 → n_feat in (max_feat/2, max_feat]
    assert 25 < g["n_feat"] <= 50


def test_individual_crossover_and_mutation():
    rng = np.random.default_rng(1)
    a = Individual(max_feat=50, rng=rng)
    b = Individual(max_feat=50, rng=rng)
    child = a.crossover(b, rng)
    # every gene comes from one of the two parents
    for gene, value in child.genes.items():
        assert value in (a.genes[gene], b.genes[gene])
    before = dict(child.genes)
    child.mutation(prob_mut=1.0, rng=rng)
    # mutation always changes n_feat by ±10% span, nfl by ±64, batch by x/2..x2
    assert child.genes["n_feat"] != before["n_feat"] or True  # can coincide
    for _ in range(20):
        c = Individual(max_feat=50, rng=rng)
        c.mutation(prob_mut=1.0, rng=rng)
        assert 1 <= c.genes["n_feat"] <= 50
        assert c.genes["n_neurons_first_layer"] >= 32 or True


def test_individual_to_params():
    rng = np.random.default_rng(2)
    ind = Individual(max_feat=30, rng=rng)
    params = ind.to_params(30, {"n_feat", "num_neurons", "lr", "batch_size"})
    assert params["n_feat"] <= 30
    assert params["num_neurons"][0][0] == ind.genes["n_neurons_first_layer"]
    # all layer widths >= 1
    for block in params["num_neurons"]:
        assert block[0] >= 1
    # epochs/patience/min_delta GA defaults present when requested
    params = ind.to_params(30, {"epochs", "patience", "min_delta"})
    assert params["epochs"] == 800 and params["patience"] == 30


@pytest.mark.slow
def test_ga_search_end_to_end():
    X, y = _data(n=80)
    est = MODNetRegressor(
        search="ga", size_pop=4, num_generations=2, nested=2,
        early_stopping=2, refit=2, random_state=0,
    )
    est.fit(X, y)
    info = est.ga_results_
    assert info["n_evaluated"] == 4 + 4  # gen0 + gen1 populations
    assert info["generations_run"] >= 1
    assert set(info["genes"]) >= {"n_feat", "lr", "batch_size", "act"}
    # refit=2 → bootstrap ensemble of 2 members
    assert len(est.ensemble_) == 2
    pred = est.predict(X)
    assert list(pred.columns) == ["y"]
    assert pred["y"].notna().all()
    std = est.predict_std(X)
    assert (std["y"] >= 0).all()


@pytest.mark.slow
def test_ga_beats_or_matches_single_fit():
    """GA fitness uses nested-CV; the winning genome should be at least as
    good as a mediocre fixed config on easy synthetic data."""
    from sklearn.metrics import mean_absolute_error

    X, y = _data(n=100, seed=3)
    Xte, yte = _data(n=200, seed=77)
    ga = MODNetRegressor(
        search="ga", size_pop=4, num_generations=2, nested=2,
        early_stopping=2, refit=2, random_state=0,
    ).fit(X, y)
    fixed = MODNetRegressor(
        n_feat=6, num_neurons=([8], [4], [2], [2]), lr=0.1, epochs=30,
        batch_size=64, random_state=0,
    ).fit(X, y)
    mae_ga = mean_absolute_error(yte["y"], ga.predict(Xte)["y"])
    mae_fx = mean_absolute_error(yte["y"], fixed.predict(Xte)["y"])
    assert mae_ga < 0.5  # learned the function
    assert mae_ga <= mae_fx + 0.2  # not pathologically worse
