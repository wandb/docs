---
title: 実験管理で CSV ファイルを追跡する
description: W&B にデータをインポートしてログする方法
menu:
  default:
    identifier: ja-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B Python ライブラリを使用して、CSV ファイルをログし、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})で可視化します。W&B ダッシュボードは、機械学習モデルからの結果を整理し可視化する中心的な場所です。これは、W&B にログされていない[以前の機械学習実験の情報を含む CSV ファイル]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ja" >}})や[データセットを含む CSV ファイル]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ja" >}})がある場合に特に便利です。

## データセットの CSV ファイルをインポートしてログする

W&B Artifacts を使用することをお勧めします。CSV ファイルの内容を再利用しやすくするためです。

1. まず、CSV ファイルをインポートします。以下のコードスニペットでは、`iris.csv` ファイル名をあなたの CSV ファイル名に置き換えてください:

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV ファイルを W&B Table に変換し、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})を利用します。

```python
# DataFrame を W&B Table に変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifact を作成し、テーブルを Artifact に追加します:

```python
# テーブルを Artifact に追加し、行制限を 200,000 に増やし、再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# データを保存するために、生の CSV ファイルを Artifact 内にログする
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts についての詳細は、[Artifacts チャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}})を参照してください。

4. 最後に、`wandb.init` を使用して W&B で追跡しログするために新しい W&B Run を開始します:

```python
# データをログするために W&B run を開始
run = wandb.init(project="tables-walkthrough")

# テーブルをログして run で可視化
run.log({"iris": iris_table})

# そして行制限を増やすためにアーティファクトとしてログ!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API は新しいバックグラウンドプロセスを開始し、データを Run にログし、デフォルトで wandb.ai に同期します。W&B ワークスペースダッシュボードでライブの可視化を表示します。以下の画像はコードスニペットのデモの出力を示しています。

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV ファイルが W&B ダッシュボードにインポートされた" >}}

以下は、前述のコードスニペットを含む完全なスクリプトです:

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame を W&B Table に変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# テーブルを Artifact に追加し、行制限を 200,000 に増やし、再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# データを保存するために、生の CSV ファイルを Artifact 内にログする
iris_table_artifact.add_file("iris.csv")

# データをログするために W&B run を開始
run = wandb.init(project="tables-walkthrough")

# テーブルをログして run で可視化
run.log({"iris": iris_table})

# そして行制限を増やすためにアーティファクトとしてログ!
run.log_artifact(iris_table_artifact)

# run を終了する (ノートブックで便利)
run.finish()
```

## 実験の CSV をインポートしてログする

場合によっては、実験の詳細が CSV ファイルにあることがあります。そのような CSV ファイルに共通する詳細には次のようなものがあります:

* 実験 run の名前
* 初期の[ノート]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ja" >}})
* 実験を区別するための[タグ]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})
* 実験に必要な設定（[Sweeps ハイパーパラメータチューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}})の利用の利点があります）。

| 実験         | モデル名          | ノート                                           | タグ          | 層の数     | 最終トレイン精度 | 最終評価精度 | トレーニング損失                             |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| 実験 1       | mnist-300-layers | トレーニングデータに過剰適合                     | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 実験 2       | mnist-250-layers | 現行の最良モデル                                | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 実験 3       | mnist-200-layers | ベースラインモデルより悪かったため、デバッグ必要 | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| 実験 N       | mnist-X-layers   | ノート                                           | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B は実験の CSV ファイルを受け取り、W&B 実験 Run に変換することができます。次のコードスニペットとコードスクリプトで、実験の CSV ファイルをインポートしてログする方法を示しています:

1. 最初に、CSV ファイルを読み込んで Pandas DataFrame に変換します。`"experiments.csv"` を CSV ファイル名に置き換えてください:

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

# 作業を容易にするための Pandas DataFrame のフォーマット
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

2. 次に、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}})を使用して W&B で追跡し、ログするための新しい W&B Run を開始します:

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

実験が進行するにつれて、メトリクスのすべてのインスタンスをログし、W&B で表示、クエリ、および分析可能にすることをお勧めするかもしれません。これを実現するには、[`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}) コマンドを使用します:

```python
run.log({key: val})
```

また、run の結果を定義するために最終的なサマリーメトリクスをオプションでログすることもできます。これを実現するには、W&B [`define_metric`]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) API を使用します。この例では、`run.summary.update()` によりサマリーメトリクスを run に追加します:

```python
run.summary.update(summaries)
```

サマリーメトリクスの詳細については、[Log Summary Metrics]({{< relref path="./log-summary.md" lang="ja" >}})を参照してください。

以下は、上記のサンプルテーブルを [W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}})に変換する完全な例のスクリプトです:

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