---
description: W&Bへのデータのインポートとログの作成
displayed_sidebar: default
---


# CSVファイルからデータと実験をログする

<head>
  <title>W&B ExperimentsでCSVファイルをトラッキング</title>
</head>

W&B Python Libraryを使用してCSVファイルをログし、それを [W&B ダッシュボード](../app.md) で可視化します。W&B ダッシュボードは、機械学習モデルの結果を整理し、可視化するための中心的な場所です。特に、[以前の機械学習実験の情報が含まれているCSVファイル](#import-and-log-your-csv-of-experiments)をW&Bにログしていない場合や、[データセットが含まれているCSVファイル](#import-and-log-your-dataset-csv-file)を持っている場合に便利です。

## データセットCSVファイルのインポートとログ

CSVファイルの内容を再利用しやすくするために、W&B Artifactsを利用することをお勧めします。

1. まずはCSVファイルをインポートします。以下のコードスニペットで、ファイル名 `iris.csv` をあなたのCSVファイル名に置き換えてください：

```python
import wandb
import pandas as pd

# CSVファイルを新しいDataFrameに読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSVファイルをW&B Tableに変換し、[W&B ダッシュボード](../app.md) を利用できるようにします。

```python
# DataFrameをW&B Tableに変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifactを作成し、テーブルをArtifactに追加します。

```python
# テーブルをArtifactに追加し、行の制限を200000に増やし、再利用を容易にする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# データを保存するために生のcsvファイルをartifactに追加
iris_table_artifact.add_file("iris.csv")
```
W&B Artifactsについての詳細は、[Artifacts チャプター](../../artifacts/intro.md) を参照してください。

4. 最後に、新しいW&B Runを開始して`wandb.init`を使用してW&Bにデータをトラッキングしてログします：

```python
# データをログするためにW&B runを開始
run = wandb.init(project="tables-walkthrough")

# runを使用してテーブルを可視化するためにログ
run.log({"iris": iris_table})

# 行の制限を増やすためにArtifactとしてログ
run.log_artifact(iris_table_artifact)
```

`wandb.init()` APIはデータをRunにログするための新しいバックグラウンドプロセスを生成し、データをデフォルトでwandb.aiに同期します。W&B ワークスペースダッシュボードでライブビジュアライゼーションを表示します。以下の画像は、このコードスニペットのデモの出力を示しています。

![CSVファイルがW&B ダッシュボードにインポートされました](/images/track/import_csv_tutorial.png)

以下は、上記のコードスニペットを含む完全なスクリプトです：

```python
import wandb
import pandas as pd

# CSVファイルを新しいDataFrameに読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrameをW&B Tableに変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# テーブルをArtifactに追加し、行の制限を200000に増やし、再利用を容易にする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# データを保存するために生のcsvファイルをartifactに追加
iris_table_artifact.add_file("iris.csv")

# データをログするためにW&B runを開始
run = wandb.init(project="tables-walkthrough")

# runを使用してテーブルを可視化するためにログ
run.log({"iris": iris_table})

# 行の制限を増やすためにArtifactとしてログ
run.log_artifact(iris_table_artifact)

# runを終了（ノートブックで有用）
run.finish()
```

## 実験のCSVファイルをインポートしてログ

場合によっては、実験の詳細を含むCSVファイルを持っていることがあります。このようなCSVファイルに含まれる一般的な詳細は以下の通りです：

* 実験runの名前
* 初期の[ノート](../../app/features/notes.md)
* 実験を区別するための[タグ](../../app/features/tags.md)
* 実験に必要な設定（さらに、[Sweeps Hyperparameter Tuning](../../sweeps/intro.md) を利用可能）

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | トレーニングデータに大いにオーバーフィット       | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | 現在のベストモデル                                | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | ベースラインモデルより悪い結果。デバッグが必要     | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | NOTES                                            | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&Bは実験のCSVファイルを受け取って、それをW&B Experiment Runに変換できます。以下のコードスニペットとコードスクリプトは、実験のCSVファイルをインポートしてログする方法を示しています：

1. まず、CSVファイルを読み込み、それをPandas DataFrameに変換します。`"experiments.csv"`をあなたのCSVファイル名に置き換えてください：

```python
import wandb
import pandas as pd

FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "Experiment"
NOTES_COL = "Notes"
TAGS_COL = "Tags"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
METRIC_COLS = ["Training Losses"]

# Pandas DataFrameを扱いやすい形式にフォーマット
for i, row in loaded_experiment_df.iterrows():
    run_name = row[EXPERIMENT_NAME_COL]
    notes = row[NOTES_COL]
    tags = row[TAGS_COL]

    config = {}
    for config_col in CONFIG_COLS:
        config[config_col] = row[config_col]

    metrics = {}
    for metric_col in METRIC_COLS:
        metrics[metric_col] = row[metric_col]

    summaries = {}
    for summary_col in SUMMARY_COLS:
        summaries[summary_col] = row[summary_col]
```

2. 次に、[`wandb.init()`](../../../ref/python/init.md) を使用して新しいW&B Runを開始し、W&Bにトラッキングしてログします：

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

実験が進行する際、メトリクスの各インスタンスをログして、W&Bで表示、クエリ、分析ができるようにすることがあります。これを実現するために[`run.log()`](../../../ref/python/log.md) コマンドを使用します：

```python
run.log({key: val})
```

また、runの結果を定義するために最終的なサマリーメトリクスをログするオプションもあります。この例では、W&B [`define_metric`](../../../ref/python/run.md#define_metric) APIを使って、`run.summary.update()` を使用してサマリーメトリクスをrunに追加します：

```python
run.summary.update(summaries)
```

サマリーメトリクスについての詳細は、[Log Summary Metrics](./log-summary.md) を参照してください。

以下は、上記のサンプルテーブルを [W&B ダッシュボード](../app.md) に変換する完全な例のスクリプトです：

```python
FILENAME = "experiments.csv"
loaded_experiment_df = pd.read_csv(FILENAME)

PROJECT_NAME = "Converted Experiments"

EXPERIMENT_NAME_COL = "Experiment"
NOTES_COL = "Notes"
TAGS_COL = "Tags"
CONFIG_COLS = ["Num Layers"]
SUMMARY_COLS = ["Final Train Acc", "Final Val Acc"]
METRIC_COLS = ["Training Losses"]

for i, row in loaded_experiment_df.iterrows():
    run_name = row[EXPERIMENT_NAME_COL]
    notes = row[NOTES_COL]
    tags = row[TAGS_COL]

    config = {}
    for config_col in CONFIG_COLS:
        config[config_col] = row[config_col]

    metrics = {}
    for metric_col in METRIC_COLS:
        metrics[metric_col] = row[metric_col]

    summaries = {}
    for summary_col in SUMMARY_COLS:
        summaries[summary_col] = row[summary_col]

    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )

    for key, val in metrics.items():
        if isinstance(val, list):
            for _val in val:
                run.log({key: _val})
        else:
            run.log({key: val})

    run.summary.update(summaries)
    run.finish()
```