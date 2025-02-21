---
title: Track CSV files with experiments
description: W&B へのデータのインポートと ログ 記録
menu:
  default:
    identifier: ja-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B Python ライブラリを使用してCSVファイルをログに記録し、[W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で可視化します。 W&B Dashboard は、機械学習モデルの結果を整理して可視化するための中央の場所です。これは、W&B に記録されていない [以前の機械学習 Experiments の情報を含む CSV ファイル]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ja" >}}) がある場合、または [データセットを含む CSV ファイル]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ja" >}}) がある場合に特に役立ちます。

## データセットの CSV ファイルをインポートしてログに記録する

W&B Artifacts を利用して、CSV ファイルの内容をより簡単に再利用できるようにすることをお勧めします。

1. まず、CSV ファイルをインポートします。次のコードスニペットでは、`iris.csv` ファイル名を CSV ファイルの名前に置き換えます。

```python
import wandb
import pandas as pd

# Read our CSV into a new DataFrame
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV ファイルを W&B Tables に変換して、[W&B Dashboards]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を利用します。

```python
# Convert the DataFrame into a W&B Table
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifact を作成し、テーブルを Artifact に追加します。

```python
# Add the table to an Artifact to increase the row
# limit to 200000 and make it easier to reuse
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# Log the raw csv file within an artifact to preserve our data
iris_table_artifact.add_file("iris.csv")
```

W&B Artifacts の詳細については、[Artifacts のチャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。

4. 最後に、新しい W&B Run を開始して、`wandb.init` で W&B に追跡およびログを記録します。

```python
# Start a W&B run to log data
run = wandb.init(project="tables-walkthrough")

# Log the table to visualize with a run...
run.log({"iris": iris_table})

# and Log as an Artifact to increase the available row limit!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API は、Run にデータをログ記録するための新しいバックグラウンドプロセスを生成し、(デフォルトで) データを wandb.ai に同期します。W&B Workspace Dashboard でライブの 可視化 を表示します。次の画像は、コードスニペットのデモの出力を示しています。

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV file imported into W&B Dashboard" >}}

上記のコードスニペットを含む完全なスクリプトは、以下にあります。

```python
import wandb
import pandas as pd

# Read our CSV into a new DataFrame
new_iris_dataframe = pd.read_csv("iris.csv")

# Convert the DataFrame into a W&B Table
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# Add the table to an Artifact to increase the row
# limit to 200000 and make it easier to reuse
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# log the raw csv file within an artifact to preserve our data
iris_table_artifact.add_file("iris.csv")

# Start a W&B run to log data
run = wandb.init(project="tables-walkthrough")

# Log the table to visualize with a run...
run.log({"iris": iris_table})

# and Log as an Artifact to increase the available row limit!
run.log_artifact(iris_table_artifact)

# Finish the run (useful in notebooks)
run.finish()
```

## Experiments の CSV ファイルをインポートしてログに記録する

場合によっては、Experiments の詳細が CSV ファイルにある場合があります。このような CSV ファイルにある一般的な詳細は次のとおりです。

* Experiment run の名前
* 初期 [notes]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ja" >}})
* Experiments を区別するための [Tags]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})
* Experiment に必要な 設定 ([Sweeps Hyperparameter Tuning]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を利用できるという追加の利点があります)。

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | Overfit way too much on training data            | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | Current best model                               | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | Did worse than the baseline model. Need to debug | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | NOTES                                            | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B は、Experiments の CSV ファイルを取得し、それを W&B Experiment Run に変換できます。次のコードスニペットとコードスクリプトは、Experiments の CSV ファイルをインポートしてログに記録する方法を示しています。

1. まず、CSV ファイルを読み込み、Pandas DataFrame に変換します。`"experiments.csv"` を CSV ファイルの名前に置き換えます。

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

# Format Pandas DataFrame to make it easier to work with
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

2. 次に、新しい W&B Run を開始して、[`wandb.init()`]({{< relref path="/ref/python/init.md" lang="ja" >}}) で W&B に追跡およびログを記録します。

```python
run = wandb.init(
    project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
)
```

Experiment の実行中に、メトリクスのすべてのインスタンスをログに記録して、W&B で表示、クエリ、および分析できるようにすることができます。[`run.log()`]({{< relref path="/ref/python/log.md" lang="ja" >}}) コマンドを使用して、これを実現します。

```python
run.log({key: val})
```

オプションで、Run の結果を定義するために、最終的な概要メトリクスをログに記録できます。W&B [`define_metric`]({{< relref path="/ref/python/run.md#define_metric" lang="ja" >}}) API を使用して、これを実現します。この例では、`run.summary.update()` を使用して、概要メトリクスを Run に追加します。

```python
run.summary.update(summaries)
```

概要メトリクスの詳細については、[概要メトリクスのログ記録]({{< relref path="./log-summary.md" lang="ja" >}}) を参照してください。

以下は、上記のサンプルテーブルを [W&B Dashboard]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に変換する完全なサンプルスクリプトです。

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
