"""Train and evaluate a basic SVD matrix factorization model with SGD.

Example:
    python svd_gd_evaluate.py --data-file /path/to/ml-100k/u.data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


DEFAULT_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]


def load_data(data_file: Path) -> pd.DataFrame:
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    return pd.read_csv(data_file, sep="\t", header=None, names=DEFAULT_COLUMNS)


def train_svd(
    train_data: pd.DataFrame,
    k: int = 20,
    lr: float = 0.005,
    reg: float = 0.02,
    epochs: int = 20,
    seed: int = 42,
) -> tuple[dict[int, int], dict[int, int], np.ndarray, np.ndarray, float]:
    user_ids = train_data["user_id"].unique()
    item_ids = train_data["item_id"].unique()

    user_index = {int(u): i for i, u in enumerate(user_ids)}
    item_index = {int(v): i for i, v in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)

    np.random.seed(seed)
    p_factors = np.random.normal(0, 0.1, (n_users, k))
    q_factors = np.random.normal(0, 0.1, (n_items, k))

    global_mean = float(train_data["rating"].mean())

    print("开始训练SVD...")
    for epoch in range(epochs):
        for row in train_data.itertuples(index=False):
            u = int(row.user_id)
            v = int(row.item_id)
            r = float(row.rating)

            ui = user_index.get(u)
            vi = item_index.get(v)
            if ui is None or vi is None:
                continue

            pred = global_mean + float(np.dot(p_factors[ui], q_factors[vi]))
            err = r - pred

            p_old = p_factors[ui].copy()
            q_old = q_factors[vi].copy()
            p_factors[ui] += lr * (err * q_old - reg * p_old)
            q_factors[vi] += lr * (err * p_old - reg * q_old)

        if (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
            print(f"Epoch {epoch + 1}/{epochs} 完成")

    return user_index, item_index, p_factors, q_factors, global_mean


def predict_rating(
    user_id: int,
    item_id: int,
    user_index: dict[int, int],
    item_index: dict[int, int],
    p_factors: np.ndarray,
    q_factors: np.ndarray,
    global_mean: float,
) -> float:
    ui = user_index.get(int(user_id))
    vi = item_index.get(int(item_id))
    if ui is None or vi is None:
        return global_mean

    pred = global_mean + float(np.dot(p_factors[ui], q_factors[vi]))
    return float(np.clip(pred, 1, 5))


def evaluate(
    test_data: pd.DataFrame,
    user_index: dict[int, int],
    item_index: dict[int, int],
    p_factors: np.ndarray,
    q_factors: np.ndarray,
    global_mean: float,
    sample_size: int = 500,
    seed: int = 42,
) -> tuple[float, float, int]:
    test_filtered = test_data[
        test_data["user_id"].isin(user_index) & test_data["item_id"].isin(item_index)
    ]
    if test_filtered.empty:
        raise ValueError("No overlapping user/item pairs found between train and test sets.")

    n = min(sample_size, len(test_filtered))
    sampled = test_filtered.sample(n=n, random_state=seed)

    preds = [
        predict_rating(
            row.user_id,
            row.item_id,
            user_index,
            item_index,
            p_factors,
            q_factors,
            global_mean,
        )
        for row in sampled.itertuples(index=False)
    ]
    actuals = sampled["rating"].to_numpy(dtype=float)

    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(np.mean(np.abs(actuals - np.array(preds))))
    return rmse, mae, n


def main() -> None:
    parser = argparse.ArgumentParser(description="SVD matrix factorization evaluation on MovieLens 100k")
    parser.add_argument("--data-file", required=True, help="Path to MovieLens u.data file")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument("--k", type=int, default=20, help="Latent factors (default: 20)")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    parser.add_argument("--reg", type=float, default=0.02, help="Regularization (default: 0.02)")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (default: 20)")
    parser.add_argument("--sample-size", type=int, default=500, help="Eval sample size (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    df_raw = load_data(Path(args.data_file))
    train_data, test_data = train_test_split(df_raw, test_size=args.test_size, random_state=args.seed)

    user_index, item_index, p_factors, q_factors, global_mean = train_svd(
        train_data,
        k=args.k,
        lr=args.lr,
        reg=args.reg,
        epochs=args.epochs,
        seed=args.seed,
    )

    rmse, mae, n = evaluate(
        test_data,
        user_index,
        item_index,
        p_factors,
        q_factors,
        global_mean,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(f"\n=== SVD矩阵分解（梯度下降）评估结果（测试样本 {n} 条） ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    print("\n=== 三算法最终对比 ===")
    print(f"{'算法':<20} {'RMSE':<10} {'MAE':<10}")
    print(f"{'User-Based CF':<20} {'0.8902':<10} {'0.7226':<10}")
    print(f"{'Item-Based CF':<20} {'0.9915':<10} {'0.7702':<10}")
    print(f"{'SVD':<20} {rmse:<10.4f} {mae:<10.4f}")


if __name__ == "__main__":
    main()
