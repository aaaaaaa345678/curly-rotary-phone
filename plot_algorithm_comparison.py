"""Plot and save a bar chart comparing recommender algorithm metrics.

Example:
    python plot_algorithm_comparison.py --output algorithm_comparison.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(output: Path, show: bool = False) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    algorithms = ["User-Based CF", "Item-Based CF", "SVD"]
    rmse_scores = [0.8902, 0.9915, 0.9643]
    mae_scores = [0.7226, 0.7702, 0.7553]

    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        rmse_scores,
        width,
        label="RMSE",
        color="#FF6B6B",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        mae_scores,
        width,
        label="MAE",
        color="#4ECDC4",
        edgecolor="white",
    )

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("推荐算法", fontsize=13)
    ax.set_ylabel("误差值（越低越好）", fontsize=13)
    ax.set_title("三种协同过滤算法性能对比", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, fontsize=12)
    ax.set_ylim(0, 1.2)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
    print(f"图表已保存：{output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate algorithm comparison bar chart")
    parser.add_argument(
        "--output",
        default="algorithm_comparison.png",
        help="Output PNG path (default: algorithm_comparison.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display chart window after saving (requires GUI environment)",
    )
    args = parser.parse_args()

    plot_comparison(Path(args.output), show=args.show)


if __name__ == "__main__":
    main()
