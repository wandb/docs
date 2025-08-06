---
title: 実験で CSV ファイルをトラッキングする
description: W&B へのデータのインポートとログ
menu:
  default:
    identifier: working-with-csv
    parent: log-objects-and-media
---

W&B Python ライブラリを使って CSV ファイルをログし、[W&B ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}}) で可視化しましょう。W&B ダッシュボードは、機械学習モデルから得られた結果を整理・可視化する中心的な場所です。これは特に、[今まで W&B にログしていなかった過去の機械学習実験情報が入った CSV ファイル]({{< relref "#import-and-log-your-csv-of-experiments" >}}) や、[データセットが入った CSV ファイル]({{< relref "#import-and-log-your-dataset-csv-file" >}}) をお持ちの場合に便利です。

## データセット CSV ファイルのインポートとログ

W&B Artifacts を活用することで、CSV ファイルの内容を再利用しやすくすることをおすすめします。

1. まず、CSV ファイルをインポートしましょう。下記のコードスニペット内の `iris.csv` をお手持ちの CSV ファイル名に置き換えてください。

```python
import wandb
import pandas as pd

# CSV ファイルを新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV ファイルを W&B Table に変換し、[W&B ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}}) を活用します。

```python
# DataFrame を W&B Table に変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 続いて、W&B Artifact を作成し、テーブルを Artifact に追加します。

```python
# Artifact にテーブルを追加して、行数上限を
# 200000 に増やし、再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 生の csv ファイルも artifact に追加してデータを保存
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts について詳しくは、[Artifacts チャプター]({{< relref "/guides/core/artifacts/" >}}) をご覧ください。

4. 最後に `wandb.init` で新しく W&B Run を開始し、W&B へログします。

```python
# データをログするために W&B run を開始
run = wandb.init(project="tables-walkthrough")

# run でテーブルを可視化
run.log({"iris": iris_table})

# さらに Artifact としてログし、行数上限を拡大！
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API はバックグラウンドプロセスを起動し、データを Run にログします（デフォルトで wandb.ai へ同期されます）。可視化は W&B ワークスペースダッシュボードでリアルタイムに見ることができます。下記イメージはコードスニペットの出力例です。

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV ファイルが W&B ダッシュボードにインポートされた例" >}}

上記のコードスニペットをまとめた全体スクリプトを以下に示します。

```python
import wandb
import pandas as pd

# CSV ファイルを新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame を W&B Table に変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# Artifact にテーブルを追加して、行数上限を
# 200000 に増やし、再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 生の csv ファイルも artifact に追加してデータを保存
iris_table_artifact.add_file("iris.csv")

# データをログするために W&B run を開始
run = wandb.init(project="tables-walkthrough")

# run でテーブルを可視化
run.log({"iris": iris_table})

# さらに Artifact としてログし、行数上限を拡大！
run.log_artifact(iris_table_artifact)

# Run を終了（ノートブックで便利）
run.finish()
```

## 実験管理用 CSV ファイルのインポートとログ

場合によっては、実験の詳細情報が CSV ファイルにまとまっていることもあります。こうした CSV によく含まれている情報の例は以下の通りです。

* 実験 run の名前
* [ノート]({{< relref "/guides/models/track/runs/#add-a-note-to-a-run" >}})
* 複数の実験を区別するための [タグ]({{< relref "/guides/models/track/runs/tags.md" >}})
* 実験に必要な設定（[Sweeps ハイパーパラメータチューニング]({{< relref "/guides/models/sweeps/" >}})にも活用可能）

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | トレーニングデータに過学習しすぎ                 | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | 現在のベストモデル                              | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | ベースラインモデルより性能が低い。要デバッグ      | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | ノート                                          | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B では、実験用の CSV ファイルを W&B Experiment Run に変換できます。下記のコードスニペットおよびスクリプトで、実験情報の CSV ファイルをインポートしログする方法を実演します。

1. まず、CSV ファイルを読み込み Pandas DataFrame に変換します。`"experiments.csv"` を使用しているファイル名に置き換えてください。

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

# 扱いやすくするため DataFrame を整形
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

2. 次に、[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) を使って新しい W&B Run を開始し、W&B にログします。

    ```python
    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )
    ```

実験ラン中に、各メトリクスをすべてログしたい場合は [`run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}) コマンドを使ってください。

```python
run.log({key: val})
```

run の結果（サマリーメトリクス）を [`define_metric`]({{< relref "/ref/python/sdk/classes/run#define_metric" >}}) API で登録することも可能です。この例では `run.summary.update()` を使ってサマリーメトリクスを run に追加しています。

```python
run.summary.update(summaries)
```

サマリーメトリクスについては [サマリーメトリクスをログする方法]({{< relref "./log-summary.md" >}}) もご参照ください。

上記のサンプルテーブルを [W&B ダッシュボード]({{< relref "/guides/models/track/workspaces.md" >}}) に変換する全体サンプルスクリプトはこちらです。

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