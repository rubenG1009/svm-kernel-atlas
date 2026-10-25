"""Small, reproducible SVM kernel experiment on synthetic 2D data."""

import argparse
import json
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles, make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, laplacian_kernel, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 42
KERNELS = ("linear", "rbf", "poly", "laplacian", "rational_quadratic", "cosine")
COLORS = ("#315a9c", "#e08043")


def datasets(seed=SEED):
    """Return both data sets with an independent generator using the same seed."""
    return {
        "moons": make_moons(n_samples=500, noise=0.23, random_state=seed),
        "circles": make_circles(
            n_samples=500, noise=0.12, factor=0.45, random_state=seed
        ),
    }


def split_data(X, y, seed=SEED):
    """Split indices once so every kernel sees exactly the same observations."""
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )
    return train_idx, test_idx


def rational_quadratic_kernel(X, Y, gamma=1.0):
    """Multi-scale kernel with alpha=1 and gamma=1/(2 * length_scale**2)."""
    squared_distance = euclidean_distances(X, Y, squared=True)
    return 1.0 / (1.0 + gamma * squared_distance)


def fit_model(X, y, train_idx, test_idx, kernel, c=1.0, gamma=1.0, degree=3):
    kernel_impl = {
        "laplacian": partial(laplacian_kernel, gamma=gamma),
        "rational_quadratic": partial(rational_quadratic_kernel, gamma=gamma),
        "cosine": cosine_similarity,
    }.get(kernel, kernel)
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel_impl, C=c, gamma=gamma, degree=degree),
    )
    model.fit(X[train_idx], y[train_idx])
    svm = model.named_steps["svc"]
    metrics = {
        "accuracy_train": float(accuracy_score(y[train_idx], model.predict(X[train_idx]))),
        "accuracy_test": float(accuracy_score(y[test_idx], model.predict(X[test_idx]))),
        "support_vectors": int(svm.n_support_.sum()),
    }
    return model, metrics


def run_experiment(seed=SEED, c=1.0, gamma=1.0, degree=3):
    """Fit the baseline models and retain indices/models for plotting and tests."""
    runs = {}
    for name, (X, y) in datasets(seed).items():
        train_idx, test_idx = split_data(X, y, seed)
        models, metrics = {}, {}
        for kernel in KERNELS:
            models[kernel], metrics[kernel] = fit_model(
                X, y, train_idx, test_idx, kernel, c, gamma, degree
            )
        runs[name] = {
            "X": X,
            "y": y,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "models": models,
            "metrics": metrics,
        }
    return runs


def plot_panel(ax, run, model, metrics, title):
    X, y = run["X"], run["y"]
    lo = X.min(axis=0) - 0.35
    hi = X.max(axis=0) + 0.35
    xx, yy = np.meshgrid(
        np.linspace(lo[0], hi[0], 180), np.linspace(lo[1], hi[1], 180)
    )
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    score = model.decision_function(grid).reshape(xx.shape)
    ax.contourf(xx, yy, score, levels=[-100, 0, 100], colors=COLORS, alpha=0.12)
    ax.contour(xx, yy, score, levels=[0], colors=["#252525"], linewidths=1.2)
    train_idx, test_idx = run["train_idx"], run["test_idx"]
    for label, color in enumerate(COLORS):
        mask_train = train_idx[y[train_idx] == label]
        mask_test = test_idx[y[test_idx] == label]
        ax.scatter(
            X[mask_train, 0], X[mask_train, 1], c=color, s=14,
            alpha=0.75, edgecolors="none", rasterized=True
        )
        ax.scatter(
            X[mask_test, 0], X[mask_test, 1], facecolors="none",
            edgecolors=color, s=34, linewidths=1.05, rasterized=True
        )
    ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="x₁", ylabel="x₂")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(
        f"{title}\ntrain {metrics['accuracy_train']:.3f} · test "
        f"{metrics['accuracy_test']:.3f} · SV {metrics['support_vectors']}",
        fontsize=10,
    )


def plot_baseline(runs, output):
    fig, axes = plt.subplots(2, len(KERNELS), figsize=(5 * len(KERNELS), 9), constrained_layout=True)
    for row, (name, run) in enumerate(runs.items()):
        for col, kernel in enumerate(KERNELS):
            plot_panel(axes[row, col], run, run["models"][kernel],
                       run["metrics"][kernel], f"{name} · {kernel}")
    fig.suptitle("SVM: misma partición y escala aprendida solo del entrenamiento\n"
                 "puntos rellenos = train · aros = test · línea = frontera", fontsize=14)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def sweep(runs, parameter, values, c, gamma, degree, output):
    kernels = KERNELS if parameter == "c" else (("rbf", "poly", "laplacian", "rational_quadratic") if parameter == "gamma" else ("poly",))
    results = {}
    for kernel in kernels:
        fig, axes = plt.subplots(2, len(values), figsize=(5 * len(values), 9),
                                 squeeze=False, constrained_layout=True)
        results[kernel] = {}
        for row, (name, run) in enumerate(runs.items()):
            results[kernel][name] = {}
            for col, value in enumerate(values):
                params = {"c": c, "gamma": gamma, "degree": degree}
                params[parameter] = value
                model, metrics = fit_model(
                    run["X"], run["y"], run["train_idx"], run["test_idx"],
                    kernel, **params
                )
                results[kernel][name][str(value)] = metrics
                plot_panel(axes[row, col], run, model, metrics,
                           f"{name} · {parameter}={value}")
        fig.suptitle(f"{kernel}: variar solo {parameter} · resto fijo", fontsize=14)
        fig.savefig(output / f"sweep_{parameter}_{kernel}.png", dpi=180)
        plt.close(fig)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--vary", choices=("c", "gamma", "degree"))
    parser.add_argument("--values", nargs="+", type=float)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.c <= 0 or args.gamma <= 0 or args.degree < 1:
        parser.error("C y gamma deben ser positivos; degree debe ser >= 1")
    if bool(args.vary) != bool(args.values):
        parser.error("--vary y --values deben indicarse juntos")
    if args.values:
        if not 2 <= len(args.values) <= 4 or any(v <= 0 for v in args.values):
            parser.error("--values requiere de 2 a 4 valores positivos")
        if args.vary == "degree":
            if any(v < 1 or not v.is_integer() for v in args.values):
                parser.error("los valores de degree deben ser enteros >= 1")
            args.values = [int(v) for v in args.values]
    return args


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = run_experiment(args.seed, args.c, args.gamma, args.degree)
    plot_baseline(runs, args.output / "kernel_comparison.png")
    report = {
        "seed": args.seed,
        "samples_per_dataset": 500,
        "test_fraction": 0.3,
        "parameters": {"c": args.c, "gamma": args.gamma, "degree": args.degree},
        "baseline": {name: run["metrics"] for name, run in runs.items()},
    }
    if args.vary:
        report["sweep"] = {
            "parameter": args.vary, "values": args.values,
            "results": sweep(runs, args.vary, args.values, args.c, args.gamma,
                             args.degree, args.output),
        }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Figures and metrics: {args.output.resolve()}")


if __name__ == "__main__":
    main()
