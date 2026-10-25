"""Export the fitted SVM models to a static, inspectable dashboard dataset."""

import json
import sys
from pathlib import Path

import numpy as np
from contourpy import contour_generator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiment import SEED, datasets, fit_model, split_data  # noqa: E402


GRID_SIZE = 55
GAMMAS = (0.1, 1.0, 10.0)


def export_dataset(name, X, y):
    train_idx, test_idx = split_data(X, y, SEED)
    lo, hi = X.min(axis=0) - 0.35, X.max(axis=0) + 0.35
    xs = np.linspace(lo[0], hi[0], GRID_SIZE)
    ys = np.linspace(lo[1], hi[1], GRID_SIZE)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    split = np.full(len(y), "train", dtype=object)
    split[test_idx] = "test"
    configurations = [("linear", None), ("cosine", None)] + [
        (kernel, gamma)
        for kernel in ("rbf", "poly", "laplacian", "rational_quadratic")
        for gamma in GAMMAS
    ]
    models = {}
    for kernel, gamma in configurations:
        key = kernel if gamma is None else f"{kernel}:{gamma:g}"
        model, metrics = fit_model(
            X, y, train_idx, test_idx, kernel, c=1.0,
            gamma=1.0 if gamma is None else gamma, degree=3,
        )
        prediction = model.predict(X)
        point_scores = model.decision_function(X)
        score = model.decision_function(grid).reshape(xx.shape)
        contours = contour_generator(x=xs, y=ys, z=score).lines(0.0)
        support = set(train_idx[model.named_steps["svc"].support_].tolist())
        points = [
            {
                "id": int(i),
                "x": round(float(X[i, 0]), 5),
                "y": round(float(X[i, 1]), 5),
                "true": int(y[i]),
                "pred": int(prediction[i]),
                "score": round(float(point_scores[i]), 5),
                "split": split[i],
                "support": i in support,
            }
            for i in range(len(y))
        ]
        confusion = [
            [int(np.sum((y[test_idx] == true) &
                        (prediction[test_idx] == pred))) for pred in (0, 1)]
            for true in (0, 1)
        ]
        models[key] = {
            "kernel": kernel,
            "gamma": gamma,
            "metrics": {
                **metrics,
                "correct_test": int(np.sum(prediction[test_idx] == y[test_idx])),
                "test_total": len(test_idx),
                "confusion": confusion,
            },
            "scoreGrid": np.round(score, 5).tolist(),
            "boundary": [np.round(line, 5).tolist() for line in contours],
            "points": points,
        }
    return {
        "name": name,
        "xGrid": np.round(xs, 5).tolist(),
        "yGrid": np.round(ys, 5).tolist(),
        "bounds": [np.round(lo, 5).tolist(), np.round(hi, 5).tolist()],
        "models": models,
    }


def main():
    data = {
        "source": "experiment.py",
        "seed": SEED,
        "samplesPerDataset": 500,
        "trainSize": 350,
        "testSize": 150,
        "parameters": {"C": 1, "degree": 3, "rationalQuadraticAlpha": 1},
        "scoreClip": 2.2,
        "floorZ": -2.65,
        "datasets": {
            name: export_dataset(name, X, y)
            for name, (X, y) in datasets(SEED).items()
        },
    }
    output = Path(__file__).parent / "data.json"
    payload = json.dumps(data, separators=(",", ":"))
    output.write_text(payload, encoding="utf-8")
    # A script wrapper lets index.html also open directly from the filesystem.
    output.with_suffix(".js").write_text(
        "window.SVM_DATA = " + payload + ";\n", encoding="utf-8"
    )
    print(f"Wrote {output} ({output.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
