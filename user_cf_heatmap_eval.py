"""User-based CF utilities for MovieLens 100k.

Features:
- Build a small user-similarity heatmap and save as PNG.
- Evaluate User-Based CF with RMSE/MAE on a test split.

Example:
    python user_cf_heatmap_eval.py --data-file /path/to/ml-100k/u.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_data(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return pd.read_csv(data_file, sep="\t", header=None, names=COLUMNS)


def build_user_item_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="user_id", columns="item_id", values="rating").fillna(0)


def save_similarity_heatmap(
    matrix: pd.DataFrame,
    output: Path,
    max_users: int = 50,
    max_items: int = 100,
) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    small_matrix = matrix.iloc[:max_users, :max_items]
    sim = cosine_similarity(small_matrix)
    sim_df = pd.DataFrame(sim, index=small_matrix.index, columns=small_matrix.index)

    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_df, cmap="YlOrRd", vmin=0, vmax=1)
    plt.title("MovieLens 用户相似度热力图（前50用户）", fontsize=14)
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)
    plt.close()


def predict_rating(user_id: int, item_id: int, train_matrix: pd.DataFrame, sim_df: pd.DataFrame) -> float:
    global_mean = float(train_matrix.values.mean())

    if user_id not in train_matrix.index or item_id not in train_matrix.columns:
        return global_mean

    sim_scores = sim_df[user_id].drop(user_id)
    item_ratings = train_matrix[item_id]
    rated = item_ratings[item_ratings > 0]

    if rated.empty:
        return global_mean

    common = sim_scores.index.intersection(rated.index)
    if common.empty:
        return global_mean

    weights = sim_scores[common]
    ratings = rated[common]

    denom = weights.abs().sum()
    if denom == 0:
        return float(ratings.mean())

    pred = float(np.dot(weights, ratings) / denom)
    return float(np.clip(pred, 1, 5))


def evaluate_user_cf(df_raw: pd.DataFrame, test_size: float = 0.2, sample_size: int = 500, seed: int = 42) -> tuple[float, float, int]:
    train_data, test_data = train_test_split(df_raw, test_size=test_size, random_state=seed)

    train_matrix = build_user_item_matrix(train_data)
    sim_matrix = cosine_similarity(train_matrix)
    sim_df = pd.DataFrame(sim_matrix, index=train_matrix.index, columns=train_matrix.index)

    n = min(sample_size, len(test_data))
    test_sample = test_data.sample(n=n, random_state=seed)

    predictions = []
    actuals = []
    for row in test_sample.itertuples(index=False):
        pred = predict_rating(int(row.user_id), int(row.item_id), train_matrix, sim_df)
        predictions.append(pred)
        actuals.append(float(row.rating))

    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    mae = float(np.mean(np.abs(np.array(actuals) - np.array(predictions))))
    return rmse, mae, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate user similarity heatmap and evaluate User-Based CF")
    parser.add_argument("--data-file", required=True, help="Path to MovieLens u.data file")
    parser.add_argument("--heatmap-output", default="movielens_heatmap.png", help="Path to output heatmap PNG")
    parser.add_argument("--max-users", type=int, default=50, help="Users shown in heatmap (default: 50)")
    parser.add_argument("--max-items", type=int, default=100, help="Items used in heatmap sim calc (default: 100)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--sample-size", type=int, default=500, help="Evaluation sample size (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    df_raw = load_data(Path(args.data_file))
    print(f"数据总条数：{len(df_raw)}")
    print(f"用户数量：{df_raw['user_id'].nunique()}")
    print(f"电影数量：{df_raw['item_id'].nunique()}")

    matrix = build_user_item_matrix(df_raw)
    print(f"评分矩阵大小：{matrix.shape}")

    save_similarity_heatmap(
        matrix,
        output=Path(args.heatmap_output),
        max_users=args.max_users,
        max_items=args.max_items,
    )
    print(f"热力图已保存：{args.heatmap_output}")

    rmse, mae, n = evaluate_user_cf(
        df_raw,
        test_size=args.test_size,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(f"\n=== User-Based CF 评估结果（测试样本 {n} 条） ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")


if __name__ == "__main__":
    main()
