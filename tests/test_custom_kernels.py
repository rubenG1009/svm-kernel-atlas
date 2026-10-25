"""Mathematical and export checks for the additional SVM kernels."""

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.build_data import export_dataset  # noqa: E402
from experiment import datasets, fit_model, rational_quadratic_kernel, split_data  # noqa: E402


def test_custom_kernel_formulas_and_gram_matrices():
    X = np.array([[1.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(
        rational_quadratic_kernel(X, X, gamma=0.5)[0, 1],
        1.0 / (1.0 + 0.5 * 5.0),
    )

    for X, y in datasets().values():
        train_idx, test_idx = split_data(X, y)
        for kernel in ("laplacian", "rational_quadratic", "cosine"):
            model, _ = fit_model(X, y, train_idx, test_idx, kernel, gamma=0.5)
            scaled = model.named_steps["standardscaler"].transform(X[train_idx[:60]])
            gram = model.named_steps["svc"].kernel(scaled, scaled)
            np.testing.assert_allclose(gram, gram.T, atol=1e-12)
            assert np.isfinite(gram).all()
            assert np.linalg.eigvalsh(gram).min() >= -1e-9
            if kernel == "laplacian":
                expected = np.exp(-0.5 * np.abs(scaled[:, None] - scaled[None, :]).sum(axis=2))
            elif kernel == "rational_quadratic":
                distance_sq = ((scaled[:, None] - scaled[None, :]) ** 2).sum(axis=2)
                expected = 1.0 / (1.0 + 0.5 * distance_sq)
            else:
                norms = np.linalg.norm(scaled, axis=1)
                expected = (scaled @ scaled.T) / np.outer(norms, norms)
            np.testing.assert_allclose(gram, expected, atol=1e-12)


def test_dashboard_export_matches_current_models():
    exported = json.loads((ROOT / "dashboard" / "data.json").read_text(encoding="utf-8"))
    javascript = (ROOT / "dashboard" / "data.js").read_text(encoding="utf-8")
    assert javascript.startswith("window.SVM_DATA = ")
    assert json.loads(javascript.removeprefix("window.SVM_DATA = ").removesuffix(";\n")) == exported
    report = json.loads((ROOT / "output" / "metrics.json").read_text(encoding="utf-8"))
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    selected_kernels = set(re.findall(r'data-kernel="([a-z_]+)"', html))
    assert selected_kernels == {"linear", "rbf", "poly", "laplacian", "rational_quadratic", "cosine"}
    for name, (X, y) in datasets().items():
        current = export_dataset(name, X, y)
        assert current == exported["datasets"][name]
        assert len(current["models"]) == 14
        for kernel, metrics in report["baseline"][name].items():
            key = kernel if kernel in ("linear", "cosine") else f"{kernel}:1"
            assert metrics == {field: current["models"][key]["metrics"][field] for field in metrics}
        for model in current["models"].values():
            confusion = model["metrics"]["confusion"]
            assert sum(map(sum, confusion)) == 150
            assert confusion[0][0] + confusion[1][1] == model["metrics"]["correct_test"]
            assert sum(point["support"] for point in model["points"] if point["split"] == "train") == model["metrics"]["support_vectors"]
