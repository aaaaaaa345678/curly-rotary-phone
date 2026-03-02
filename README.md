# curly-rotary-phone

Utilities and experiments.

## MovieLens 100k -> MySQL importer

Added script: `import_ml100k_to_mysql.py`

### Install dependencies

```bash
pip install pandas pymysql
```

### Run

```bash
export MYSQL_PASSWORD='your_password'
python import_ml100k_to_mysql.py --data-dir /path/to/ml-100k
```

Optional env vars:

- `MYSQL_HOST` (default `127.0.0.1`)
- `MYSQL_PORT` (default `3306`)
- `MYSQL_USER` (default `root`)
- `MYSQL_PASSWORD` (default empty)
- `MYSQL_DATABASE` (default `recsys`)

## 算法对比图生成

新增脚本：`plot_algorithm_comparison.py`

```bash
pip install matplotlib numpy
python plot_algorithm_comparison.py --output algorithm_comparison.png
```

可选参数：

- `--output`：输出 PNG 路径（默认 `algorithm_comparison.png`）
- `--show`：保存后弹窗显示图表（需要图形界面）

## SVD 训练与评估（梯度下降）

新增脚本：`svd_gd_evaluate.py`

```bash
pip install pandas numpy scikit-learn
python svd_gd_evaluate.py --data-file /path/to/ml-100k/u.data
```

可选参数示例：

```bash
python svd_gd_evaluate.py \
  --data-file /path/to/ml-100k/u.data \
  --k 20 --lr 0.005 --reg 0.02 --epochs 20 --sample-size 500
```


## User-Based CF 热力图与评估

新增脚本：`user_cf_heatmap_eval.py`

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python user_cf_heatmap_eval.py --data-file /path/to/ml-100k/u.data
```

可选参数：

- `--heatmap-output`：热力图输出路径（默认 `movielens_heatmap.png`）
- `--max-users` / `--max-items`：热力图计算的数据子集大小
- `--test-size` / `--sample-size` / `--seed`：评估拆分与采样参数


## Item-Based CF 评估

新增脚本：`item_cf_evaluate.py`

```bash
pip install pandas numpy scikit-learn
python item_cf_evaluate.py --data-file /path/to/ml-100k/u.data
```

可选参数：

- `--test-size` / `--sample-size` / `--seed`：评估拆分与采样参数
- `--top-k`：预测时使用的相似物品数量（默认 `10`）

