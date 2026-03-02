"""Import MovieLens 100k dataset into a MySQL recsys schema.

Usage:
  python import_ml100k_to_mysql.py --data-dir /path/to/ml-100k

Connection settings can be overridden with environment variables:
  MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import pymysql


def get_connection() -> pymysql.connections.Connection:
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "recsys"),
        charset="utf8mb4",
        autocommit=False,
    )


def import_users(cursor: pymysql.cursors.Cursor, data_dir: Path) -> int:
    users = pd.read_csv(
        data_dir / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    values = [
        (int(row.user_id), int(row.age), str(row.gender), str(row.occupation))
        for row in users.itertuples(index=False)
    ]
    cursor.executemany(
        """
        INSERT IGNORE INTO users (user_id, age, gender, occupation)
        VALUES (%s, %s, %s, %s)
        """,
        values,
    )
    return len(values)


def import_movies(cursor: pymysql.cursors.Cursor, data_dir: Path) -> int:
    movies = pd.read_csv(data_dir / "u.item", sep="|", encoding="latin-1", header=None)
    values = [(int(row[0]), str(row[1]), str(row[2])) for row in movies.itertuples(index=False)]
    cursor.executemany(
        """
        INSERT IGNORE INTO movies (movie_id, title, release_date)
        VALUES (%s, %s, %s)
        """,
        values,
    )
    return len(values)


def import_ratings(cursor: pymysql.cursors.Cursor, data_dir: Path, batch_size: int = 5000) -> int:
    ratings = pd.read_csv(
        data_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    batch: list[tuple[int, int, int, int]] = []
    inserted = 0
    for row in ratings.itertuples(index=False):
        batch.append((int(row.user_id), int(row.item_id), int(row.rating), int(row.timestamp)))
        if len(batch) >= batch_size:
            cursor.executemany(
                """
                INSERT IGNORE INTO ratings (user_id, movie_id, rating, timestamp)
                VALUES (%s, %s, %s, %s)
                """,
                batch,
            )
            inserted += len(batch)
            batch.clear()

    if batch:
        cursor.executemany(
            """
            INSERT IGNORE INTO ratings (user_id, movie_id, rating, timestamp)
            VALUES (%s, %s, %s, %s)
            """,
            batch,
        )
        inserted += len(batch)

    return inserted


def main() -> None:
    parser = argparse.ArgumentParser(description="Import MovieLens 100k to MySQL")
    parser.add_argument("--data-dir", required=True, help="Path to ml-100k directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    required_files = ["u.user", "u.item", "u.data"]
    missing = [f for f in required_files if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files under {data_dir}: {', '.join(missing)}")

    print("开始导入数据...")

    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            user_count = import_users(cursor, data_dir)
            conn.commit()
            print(f"✅ 用户数据导入完成：{user_count} 条")

            movie_count = import_movies(cursor, data_dir)
            conn.commit()
            print(f"✅ 电影数据导入完成：{movie_count} 条")

            rating_count = import_ratings(cursor, data_dir)
            conn.commit()
            print(f"✅ 评分数据导入完成：{rating_count} 条")

        print("\n🎉 所有数据导入完成！")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
