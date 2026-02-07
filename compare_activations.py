#!/usr/bin/env python3
import argparse
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


def custom_activation(x: Array) -> Array:
    # Original example: Swish + small tanh term (smooth and non-monotonic around 0)
    return x * sigmoid(x) + 0.1 * np.tanh(x)


def custom_activation_grad(x: Array) -> Array:
    s = sigmoid(x)
    t = np.tanh(x)
    return s + x * s * (1.0 - s) + 0.1 * (1.0 - t * t)


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
    args = parser.parse_args()

    acts = [
        Activation("relu", relu, relu_grad),
        Activation("tanh", tanh, tanh_grad),
        Activation("sigmoid", sigmoid, sigmoid_grad),
        Activation("custom", custom_activation, custom_activation_grad),
    ]

    x, y = make_xor_data(args.samples, args.noise, args.seed)
    split = int(len(x) * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    init_state = init_params(input_dim=2, hidden_dim=args.hidden, seed=args.seed + 1)

    results: Dict[str, Dict[str, List[float]]] = {}
    for act in acts:
        results[act.name] = train_one(
            activation=act,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            init_state=init_state,
            epochs=args.epochs,
            lr=args.lr,
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plot_losses(results, args.out)

    print("=== Activation Comparison Summary ===")
    print(f"epochs={args.epochs}, lr={args.lr}, hidden={args.hidden}, samples={args.samples}")
    print(f"loss plot saved to: {args.out}")
    print("")
    print(
        "activation | final_train_loss | final_test_loss | final_test_acc | "
        "epoch_to_acc_threshold | epoch_to_95pct_loss_improve"
    )
    print("-" * 120)

    for name, hist in results.items():
        ep = first_epoch_reaching(hist["test_acc"], args.acc_threshold)
        ep_loss = epoch_to_fractional_improvement(hist["test_loss"], fraction=0.95)
        ep_str = str(ep) if ep is not None else "not reached"
        ep_loss_str = str(ep_loss) if ep_loss is not None else "not reached"
        print(
            f"{name:9s} | {hist['train_loss'][-1]:16.4f} | {hist['test_loss'][-1]:15.4f} |"
            f" {hist['test_acc'][-1]:14.4f} | {ep_str:22s} | {ep_loss_str}"
        )


if __name__ == "__main__":
    main()
