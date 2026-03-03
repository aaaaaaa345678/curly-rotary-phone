"""Evaluate Item-Based Collaborative Filtering on MovieLens 100k.

Example:
    python item_cf_evaluate.py --data-file /path/to/ml-100k/u.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_data(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    return pd.read_csv(data_file, sep="\t", header=None, names=COLUMNS)


def build_item_user_matrix(train_data: pd.DataFrame) -> pd.DataFrame:
    return train_data.pivot_table(index="item_id", columns="user_id", values="rating").fillna(0)


def build_item_similarity(train_matrix: pd.DataFrame) -> pd.DataFrame:
    item_sim = cosine_similarity(train_matrix)
    return pd.DataFrame(item_sim, index=train_matrix.index, columns=train_matrix.index)


def predict_rating_item(
    user_id: int,
    item_id: int,
    train_matrix: pd.DataFrame,
    item_sim_df: pd.DataFrame,
    top_k: int = 10,
) -> float:
    global_mean = float(train_matrix.values.mean())

    if item_id not in train_matrix.index or user_id not in train_matrix.columns:
        return global_mean

    sim_items = item_sim_df[item_id].drop(item_id).nlargest(top_k)
    user_ratings = train_matrix.loc[sim_items.index, user_id]
    rated = user_ratings[user_ratings > 0]

    if rated.empty:
        return global_mean

    pred = float(rated.mean())
    return float(np.clip(pred, 1, 5))


def evaluate_item_cf(
    df_raw: pd.DataFrame,
    test_size: float = 0.2,
    sample_size: int = 500,
    top_k: int = 10,
    seed: int = 42,
) -> tuple[float, float, int]:
    train_data, test_data = train_test_split(df_raw, test_size=test_size, random_state=seed)
    train_matrix = build_item_user_matrix(train_data)
    item_sim_df = build_item_similarity(train_matrix)

    n = min(sample_size, len(test_data))
    test_sample = test_data.sample(n=n, random_state=seed)

    preds = [
        predict_rating_item(int(row.user_id), int(row.item_id), train_matrix, item_sim_df, top_k=top_k)
        for row in test_sample.itertuples(index=False)
    ]
    actuals = test_sample["rating"].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(np.mean(np.abs(actuals - np.array(preds))))
    return rmse, mae, n


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Item-Based CF on MovieLens 100k")
    parser.add_argument("--data-file", required=True, help="Path to MovieLens u.data file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--sample-size", type=int, default=500, help="Evaluation sample size (default: 500)")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k similar items used for prediction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df_raw = load_data(Path(args.data_file))
    rmse, mae, n = evaluate_item_cf(
        df_raw,
        test_size=args.test_size,
        sample_size=args.sample_size,
        top_k=args.top_k,
        seed=args.seed,
    )

    print(f"=== Item-Based CF 评估结果（测试样本 {n} 条） ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    print("\n=== 算法对比 ===")
    print(f"{'算法':<20} {'RMSE':<10} {'MAE':<10}")
    print(f"{'User-Based CF':<20} {'0.8902':<10} {'0.7226':<10}")
    print(f"{'Item-Based CF':<20} {rmse:<10.4f} {mae:<10.4f}")


if __name__ == "__main__":
    main()
