import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment import KERNELS, run_experiment  # noqa: E402


def test_reproducible_data_splits_predictions_and_metrics():
    first = run_experiment()
    second = run_experiment()
    for name in first:
        a, b = first[name], second[name]
        np.testing.assert_array_equal(a["X"], b["X"])
        np.testing.assert_array_equal(a["train_idx"], b["train_idx"])
        np.testing.assert_array_equal(a["test_idx"], b["test_idx"])
        assert a["metrics"] == b["metrics"]
        assert len(a["test_idx"]) == 150
        for kernel in KERNELS:
            np.testing.assert_array_equal(
                a["models"][kernel].predict(a["X"][a["test_idx"]]),
                b["models"][kernel].predict(b["X"][b["test_idx"]]),
            )


def test_scaler_is_fitted_only_on_training_rows_and_split_is_shared():
    for run in run_experiment().values():
        X, y = run["X"], run["y"]
        train_idx, test_idx = run["train_idx"], run["test_idx"]
        assert set(train_idx).isdisjoint(test_idx)
        assert sorted(np.r_[train_idx, test_idx].tolist()) == list(range(len(y)))
        assert np.bincount(y[train_idx]).tolist() == [175, 175]
        assert np.bincount(y[test_idx]).tolist() == [75, 75]
        for kernel in KERNELS:
            model = run["models"][kernel]
            scaler = model.named_steps["standardscaler"]
            assert scaler.n_samples_seen_ == len(train_idx)
            np.testing.assert_allclose(scaler.mean_, X[train_idx].mean(axis=0))
            np.testing.assert_allclose(scaler.var_, X[train_idx].var(axis=0))
            assert not np.allclose(scaler.mean_, X.mean(axis=0), atol=1e-4)
