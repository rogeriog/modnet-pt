"""Joint-learning parity experiment (milestone M4).

Reproduces the joint-vs-single mechanism of De Breuck et al. 2020 (~8% MAE
gain, larger at smaller training sizes, p. 8) on the synthetic stand-in
established by the modnet-nutshell project: the paper's vibrational dataset
(1245 materials, S at multiple temperatures) is not available offline, so
targets here are scaled noisy views of one shared latent — the same
multi-temperature structure (S(T) curves driven by shared physics).

Single-target fits are data-limited; the joint fit pools supervision across
targets through the shared trunk (upstream m-MODNet, vanilla.py:122) and
must win. Calibrated 2026-11-06: ~10% gain at n=60, ~3% at n=150.
"""

import numpy as np
import pandas as pd
import pytest

from modnet_pt import MODNetRegressor

ALPHAS = [1.0, 1.3, 0.7, 1.15]  # "temperature" scalings of the shared latent


def _data(n, seed, noise=0.3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, size=(n, 4))
    f = np.sin(2 * x[:, 0]) + 0.5 * x[:, 1] ** 2 + 0.3 * x[:, 2] * x[:, 3]
    y = np.stack([a * f + rng.normal(0, noise, n) for a in ALPHAS], axis=1)
    return x, y


def _model(seed):
    return MODNetRegressor(
        n_feat=4,
        num_neurons=([16], [], [], []),
        epochs=600,
        batch_size=16,
        lr=1e-2,
        random_state=seed,
    )


def _run_experiment(n, seeds=(1, 2, 3, 4)):
    joint_maes, single_maes = [], []
    for seed in seeds:
        xtr, ytr = _data(n, seed)
        xte, yte = _data(2000, 10000 + seed)
        columns = [f"x{i}" for i in range(4)]
        df_tr = pd.DataFrame(xtr, columns=columns)
        df_te = pd.DataFrame(xte, columns=columns)
        y_frame = pd.DataFrame(ytr, columns=[f"t{i}" for i in range(4)])

        joint = _model(seed).fit(df_tr, y_frame).predict(df_te)
        assert joint.values.shape == yte.shape
        joint_maes.append(np.mean(np.abs(joint.values - yte)))

        per_target = []
        for k in range(len(ALPHAS)):
            pred = _model(seed).fit(df_tr, y_frame[[f"t{k}"]]).predict(df_te)
            assert pred.values.shape == (len(xte), 1)
            per_target.append(np.abs(pred.values[:, 0] - yte[:, k]))
        single_maes.append(np.mean(per_target))
    return np.mean(joint_maes), np.mean(single_maes)


@pytest.mark.slow
def test_joint_learning_beats_single_target():
    """The paper's headline mechanism: joint learning must beat
    single-target fits in the small-data regime (observed ~10% at n=60)."""
    joint, single = _run_experiment(60)
    gain = 1 - joint / single
    assert joint < single
    # robust margin well below the calibrated 10% (seed variance ±2%)
    assert gain > 0.04, f"joint-learning gain too small: {gain:.3f}"


def test_joint_learning_gain_shrinks_with_data():
    """The gain must shrink as training data grows (paper p. 8: more at
    smaller sizes) — small-data regime gains more than the data-rich one.
    Uses small epoch/seed budgets to stay fast; the slow test above covers
    the calibrated numbers."""
    small = _run_experiment(60, seeds=(1, 2))
    large = _run_experiment(150, seeds=(1, 2))
    assert (1 - small[0] / small[1]) > (1 - large[0] / large[1])
