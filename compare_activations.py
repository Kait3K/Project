#!/usr/bin/env python3
import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray
EPS = 1e-8


@dataclass
class Activation:
    name: str
    forward: Callable[[Array], Array]
    grad: Callable[[Array], Array]


def sigmoid(x: Array) -> Array:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def relu(x: Array) -> Array:
    return np.maximum(0.0, x)


def relu_grad(x: Array) -> Array:
    return (x > 0.0).astype(x.dtype)


def tanh(x: Array) -> Array:
    return np.tanh(x)


def tanh_grad(x: Array) -> Array:
    t = np.tanh(x)
    return 1.0 - t * t


def sigmoid_grad(x: Array) -> Array:
    s = sigmoid(x)
    return s * (1.0 - s)


def hermite_polynomial(n: int, x: Array) -> Array:
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    h_nm2 = np.ones_like(x)
    h_nm1 = 2.0 * x
    for k in range(1, n):
        h_n = 2.0 * x * h_nm1 - 2.0 * k * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def harmonic_oscillator_eigenfunction(n: int, x: Array) -> Array:
    # Dimensionless 1D harmonic oscillator eigenfunction: psi_n(x)
    h_n = hermite_polynomial(n, x)
    norm = 1.0 / math.sqrt((2.0 ** n) * math.factorial(n) * math.sqrt(math.pi))
    return norm * h_n * np.exp(-0.5 * x * x)


def harmonic_oscillator_eigenfunction_grad(n: int, x: Array) -> Array:
    h_n = hermite_polynomial(n, x)
    if n == 0:
        h_nm1 = np.zeros_like(x)
    else:
        h_nm1 = hermite_polynomial(n - 1, x)
    norm = 1.0 / math.sqrt((2.0 ** n) * math.factorial(n) * math.sqrt(math.pi))
    # d/dx [H_n(x) exp(-x^2/2)] = (2n H_{n-1}(x) - x H_n(x)) exp(-x^2/2)
    return norm * (2.0 * n * h_nm1 - x * h_n) * np.exp(-0.5 * x * x)


def custom_activation(x: Array, n: int = 2, alpha: float = 0.75, beta: float = 1.0) -> Array:
    # Identity + scaled HO eigenfunction to keep gradients stable.
    z = beta * x
    return x + alpha * harmonic_oscillator_eigenfunction(n, z)


def custom_activation_grad(x: Array, n: int = 2, alpha: float = 0.75, beta: float = 1.0) -> Array:
    z = beta * x
    return 1.0 + alpha * beta * harmonic_oscillator_eigenfunction_grad(n, z)


def make_xor_data(n_samples: int, noise: float, seed: int) -> Tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).astype(np.float64).reshape(-1, 1)

    x += rng.normal(0.0, noise, size=x.shape)
    flip_mask = rng.random(size=(n_samples, 1)) < (noise * 0.35)
    y = np.where(flip_mask, 1.0 - y, y)

    # Standardize for stabler optimization.
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + EPS)
    return x, y


def bce_loss(y_true: Array, y_pred: Array) -> float:
    y_pred = np.clip(y_pred, EPS, 1.0 - EPS)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def accuracy(y_true: Array, y_pred: Array) -> float:
    pred = (y_pred >= 0.5).astype(np.float64)
    return float(np.mean(pred == y_true))


def init_params(input_dim: int, hidden_dim: int, seed: int) -> Dict[str, Array]:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.35, size=(input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim), dtype=np.float64),
        "W2": rng.normal(0.0, 0.35, size=(hidden_dim, 1)),
        "b2": np.zeros((1, 1), dtype=np.float64),
    }


def train_one(
    activation: Activation,
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_test: Array,
    init_state: Dict[str, Array],
    epochs: int,
    lr: float,
) -> Dict[str, List[float]]:
    w1 = init_state["W1"].copy()
    b1 = init_state["b1"].copy()
    w2 = init_state["W2"].copy()
    b2 = init_state["b2"].copy()

    train_loss_hist: List[float] = []
    test_loss_hist: List[float] = []
    train_acc_hist: List[float] = []
    test_acc_hist: List[float] = []

    n = x_train.shape[0]

    for _ in range(epochs):
        z1 = x_train @ w1 + b1
        a1 = activation.forward(z1)
        z2 = a1 @ w2 + b2
        y_hat = sigmoid(z2)

        train_loss = bce_loss(y_train, y_hat)
        train_acc = accuracy(y_train, y_hat)

        dz2 = (y_hat - y_train) / n
        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2 @ w2.T
        dz1 = da1 * activation.grad(z1)
        dw1 = x_train.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        w1 -= lr * dw1
        b1 -= lr * db1
        w2 -= lr * dw2
        b2 -= lr * db2

        test_pred = sigmoid(activation.forward(x_test @ w1 + b1) @ w2 + b2)
        test_loss = bce_loss(y_test, test_pred)
        test_acc = accuracy(y_test, test_pred)

        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
        train_acc_hist.append(train_acc)
        test_acc_hist.append(test_acc)

    return {
        "train_loss": train_loss_hist,
        "test_loss": test_loss_hist,
        "train_acc": train_acc_hist,
        "test_acc": test_acc_hist,
    }


def first_epoch_reaching(values: List[float], threshold: float) -> Optional[int]:
    for i, v in enumerate(values, start=1):
        if v >= threshold:
            return i
    return None


def epoch_to_fractional_improvement(losses: List[float], fraction: float = 0.95) -> Optional[int]:
    if not losses:
        return None
    start = losses[0]
    end = losses[-1]
    gain = start - end
    if gain <= 0.0:
        return None
    target = start - gain * fraction
    for i, value in enumerate(losses, start=1):
        if value <= target:
            return i
    return None


def plot_losses(results: Dict[str, Dict[str, List[float]]], output_path: str) -> None:
    plt.figure(figsize=(11, 6))
    for name, hist in results.items():
        plt.plot(hist["train_loss"], label=f"{name} train", linewidth=2)
        plt.plot(hist["test_loss"], label=f"{name} test", linestyle="--", linewidth=2)

    plt.title("Training/Test Loss Comparison by Activation")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid(alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def parse_seed_list(seed_list_arg: str) -> List[int]:
    if not seed_list_arg.strip():
        return []
    seeds: List[int] = []
    for token in seed_list_arg.split(","):
        token = token.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("--seed-list did not contain any valid seed")
    return seeds


def parse_int_list(int_list_arg: str, arg_name: str) -> List[int]:
    if not int_list_arg.strip():
        return []
    values: List[int] = []
    for token in int_list_arg.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError(f"{arg_name} did not contain any valid integer")
    return values


def aggregate_histories(per_seed_histories: List[Dict[str, List[float]]]) -> Dict[str, List[float]]:
    if not per_seed_histories:
        raise ValueError("per_seed_histories must not be empty")
    metrics = ["train_loss", "test_loss", "train_acc", "test_acc"]
    out: Dict[str, List[float]] = {}
    for metric in metrics:
        stacked = np.array([h[metric] for h in per_seed_histories], dtype=np.float64)
        out[metric] = np.mean(stacked, axis=0).tolist()
    return out


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.mean(arr)), float(np.std(arr))


def epoch_stats(values: List[Optional[int]]) -> Tuple[str, str]:
    reached = [v for v in values if v is not None]
    if not reached:
        return "not reached", f"0/{len(values)}"
    m, s = mean_std([float(v) for v in reached])
    return f"{m:.1f}+-{s:.1f}", f"{len(reached)}/{len(values)}"


def summarize_histories(histories: List[Dict[str, List[float]]], acc_threshold: float) -> Dict[str, object]:
    train_final = [h["train_loss"][-1] for h in histories]
    test_final = [h["test_loss"][-1] for h in histories]
    test_acc_final = [h["test_acc"][-1] for h in histories]
    epoch_to_acc = [first_epoch_reaching(h["test_acc"], acc_threshold) for h in histories]
    epoch_to_loss = [epoch_to_fractional_improvement(h["test_loss"], fraction=0.95) for h in histories]

    train_m, train_s = mean_std(train_final)
    test_m, test_s = mean_std(test_final)
    acc_m, acc_s = mean_std(test_acc_final)
    ep_acc_str, ep_acc_reach = epoch_stats(epoch_to_acc)
    ep_loss_str, ep_loss_reach = epoch_stats(epoch_to_loss)

    reached_loss = [v for v in epoch_to_loss if v is not None]
    ep_loss_mean = float(np.mean(reached_loss)) if reached_loss else float("nan")
    ep_loss_std = float(np.std(reached_loss)) if reached_loss else float("nan")

    return {
        "train_m": train_m,
        "train_s": train_s,
        "test_m": test_m,
        "test_s": test_s,
        "acc_m": acc_m,
        "acc_s": acc_s,
        "ep_acc_str": ep_acc_str,
        "ep_acc_reach": ep_acc_reach,
        "ep_loss_str": ep_loss_str,
        "ep_loss_reach": ep_loss_reach,
        "ep_loss_mean": ep_loss_mean,
        "ep_loss_std": ep_loss_std,
    }


def plot_custom_n_loss_curves(
    n_values: List[int], per_n_results: Dict[int, List[Dict[str, List[float]]]], output_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax_train, ax_test = axes

    for n in n_values:
        mean_hist = aggregate_histories(per_n_results[n])
        ax_train.plot(mean_hist["train_loss"], linewidth=2, label=f"n={n}")
        ax_test.plot(mean_hist["test_loss"], linewidth=2, label=f"n={n}")

    ax_train.set_title("Custom Activation: Train Loss Decay by n")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("BCE Loss")
    ax_train.grid(alpha=0.3)

    ax_test.set_title("Custom Activation: Test Loss Decay by n")
    ax_test.set_xlabel("Epoch")
    ax_test.set_ylabel("BCE Loss")
    ax_test.grid(alpha=0.3)
    ax_test.legend(ncol=2)

    plt.tight_layout()
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare activation functions with train/test loss curves (NumPy-only)."
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--noise", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--acc-threshold", type=float, default=0.90)
    parser.add_argument("--out", type=str, default="results/loss_comparison.png")
    parser.add_argument("--custom-n-list", type=str, default="")
    parser.add_argument("--custom-n-plot-out", type=str, default="results/custom_n_comparison.png")
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument("--custom-n", type=int, default=2)
    parser.add_argument("--custom-alpha", type=float, default=0.75)
    parser.add_argument("--custom-beta", type=float, default=1.0)
    args = parser.parse_args()
    if args.custom_n < 0:
        raise ValueError("--custom-n must be >= 0")
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be >= 1")

    explicit_seeds = parse_seed_list(args.seed_list)
    if explicit_seeds:
        seeds = explicit_seeds
    else:
        seeds = [args.seed + i for i in range(args.num_seeds)]
    custom_n_list = parse_int_list(args.custom_n_list, "--custom-n-list")
    if custom_n_list:
        if any(n < 0 for n in custom_n_list):
            raise ValueError("--custom-n-list must contain only n >= 0")
        custom_n_list = sorted(set(custom_n_list))

    acts = [
        Activation("relu", relu, relu_grad),
        Activation("tanh", tanh, tanh_grad),
        Activation("sigmoid", sigmoid, sigmoid_grad),
        Activation(
            f"custom(n={args.custom_n})",
            lambda x: custom_activation(x, n=args.custom_n, alpha=args.custom_alpha, beta=args.custom_beta),
            lambda x: custom_activation_grad(
                x, n=args.custom_n, alpha=args.custom_alpha, beta=args.custom_beta
            ),
        ),
    ]

    per_seed_results: Dict[str, List[Dict[str, List[float]]]] = {act.name: [] for act in acts}
    for seed in seeds:
        x, y = make_xor_data(args.samples, args.noise, seed)
        split = int(len(x) * 0.8)
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        init_state = init_params(input_dim=2, hidden_dim=args.hidden, seed=seed + 1)
        for act in acts:
            history = train_one(
                activation=act,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                init_state=init_state,
                epochs=args.epochs,
                lr=args.lr,
            )
            per_seed_results[act.name].append(history)

    results: Dict[str, Dict[str, List[float]]] = {
        name: aggregate_histories(histories) for name, histories in per_seed_results.items()
    }

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plot_losses(results, args.out)

    print("=== Activation Comparison Summary ===")
    print(f"epochs={args.epochs}, lr={args.lr}, hidden={args.hidden}, samples={args.samples}")
    print(f"seeds={seeds}")
    print(f"loss plot saved to: {args.out}")
    print("")
    print(
        "activation | final_train_loss(mean+-std) | final_test_loss(mean+-std) | "
        "final_test_acc(mean+-std) | epoch_to_acc_threshold(mean+-std) | "
        "reach(acc) | epoch_to_95pct_loss_improve(mean+-std) | reach(loss)"
    )
    print("-" * 200)

    for name, histories in per_seed_results.items():
        summary = summarize_histories(histories, args.acc_threshold)
        print(
            f"{name:13s} | {summary['train_m']:.4f}+-{summary['train_s']:.4f} |"
            f" {summary['test_m']:.4f}+-{summary['test_s']:.4f} |"
            f" {summary['acc_m']:.4f}+-{summary['acc_s']:.4f} |"
            f" {summary['ep_acc_str']:29s} | {summary['ep_acc_reach']:10s} |"
            f" {summary['ep_loss_str']:35s} | {summary['ep_loss_reach']}"
        )

    if custom_n_list:
        per_n_results: Dict[int, List[Dict[str, List[float]]]] = {n: [] for n in custom_n_list}
        for seed in seeds:
            x, y = make_xor_data(args.samples, args.noise, seed)
            split = int(len(x) * 0.8)
            x_train, x_test = x[:split], x[split:]
            y_train, y_test = y[:split], y[split:]

            init_state = init_params(input_dim=2, hidden_dim=args.hidden, seed=seed + 1)
            for n in custom_n_list:
                act = Activation(
                    f"custom(n={n})",
                    lambda x, n=n: custom_activation(x, n=n, alpha=args.custom_alpha, beta=args.custom_beta),
                    lambda x, n=n: custom_activation_grad(x, n=n, alpha=args.custom_alpha, beta=args.custom_beta),
                )
                per_n_results[n].append(
                    train_one(
                        activation=act,
                        x_train=x_train,
                        y_train=y_train,
                        x_test=x_test,
                        y_test=y_test,
                        init_state=init_state,
                        epochs=args.epochs,
                        lr=args.lr,
                    )
                )

        n_stats: Dict[int, Dict[str, object]] = {
            n: summarize_histories(histories, args.acc_threshold) for n, histories in per_n_results.items()
        }
        plot_custom_n_loss_curves(custom_n_list, per_n_results, args.custom_n_plot_out)

        print("")
        print("=== Custom n Sweep Summary ===")
        print(f"n values={custom_n_list}")
        print(f"custom n loss-curve plot saved to: {args.custom_n_plot_out}")
        print("n | final_test_loss(mean+-std) | final_test_acc(mean+-std) | epoch_to_95pct_loss_improve(mean+-std)")
        print("-" * 110)
        for n in custom_n_list:
            s = n_stats[n]
            print(
                f"{n:2d} | {s['test_m']:.4f}+-{s['test_s']:.4f} |"
                f" {s['acc_m']:.4f}+-{s['acc_s']:.4f} | {s['ep_loss_str']}"
            )


if __name__ == "__main__":
    main()
