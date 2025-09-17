---
title: CSV ファイルを 実験 で追跡する
description: W&B へのデータの取り込みとログ
menu:
  default:
    identifier: ja-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B の Python ライブラリを使って CSV ファイルをログし、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で可視化しましょう。W&B ダッシュボードは、機械学習 モデルの結果を整理・可視化するための中心的な場所です。W&B にログされていない過去の 実験 の情報を含む [CSV ファイル]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ja" >}}) や、[データセット を含む CSV ファイル]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ja" >}}) を扱う場合に特に便利です。

## データセットの CSV ファイルを取り込み、ログする




CSV の内容を再利用しやすくするため、W&B Artifacts を活用することをおすすめします。

1. まずは CSV ファイルを読み込みます。以下の コードスニペット では、`iris.csv` をあなたの CSV ファイル名に置き換えてください:

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV ファイルを W&B Table に変換し、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) を活用できるようにします。 

```python
# DataFrame を W&B Table に変換する
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifact を作成し、テーブルを Artifact に追加します:

```python
# テーブルを Artifact に追加して
# 行上限を 200000 に引き上げ、再利用を容易にする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# データを保持するため、生の CSV ファイルも Artifact 内にログする
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts の詳細は、[Artifacts チャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) を参照してください。  

4. 最後に、`wandb.init` で新しい W&B Run を開始し、W&B にトラッキングおよびログを行います:

```python
# W&B Run を開始してデータをログする
run = wandb.init(project="tables-walkthrough")

# Run で可視化できるようにテーブルをログする...
run.log({"iris": iris_table})

# さらに Artifact としてもログして、利用可能な行上限を引き上げる！
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API は、Run にデータをログするためのバックグラウンド プロセスを新規起動し、データをデフォルトで wandb.ai に同期します。W&B Workspace ダッシュボードでライブの可視化を確認できます。以下はコードスニペットの出力例です。

{{< img src="/images/track/import_csv_tutorial.png" alt="W&B ダッシュボードにインポートされた CSV ファイル" >}}

前述の コードスニペット をまとめたスクリプト全体は以下のとおりです:

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame を W&B Table に変換する
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# テーブルを Artifact に追加して
# 行上限を 200000 に引き上げ、再利用を容易にする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 生の CSV ファイルを Artifact 内にログしてデータを保持する
iris_table_artifact.add_file("iris.csv")

# W&B Run を開始してデータをログする
run = wandb.init(project="tables-walkthrough")

# Run で可視化できるようにテーブルをログする...
run.log({"iris": iris_table})

# さらに Artifact としてもログして、利用可能な行上限を引き上げる！
run.log_artifact(iris_table_artifact)

# Run を終了する（ノートブックで便利）
run.finish()
```

## Experiments の CSV を取り込み、ログする




実験の詳細を CSV ファイルで管理している場合があります。そうした CSV によく含まれる情報の例は次のとおりです:

* 実験の Run 名
* Run に [ノートを追加]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ja" >}}) するための初期ノート
* 実験を区別するための [タグ]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})
* 実験に必要な 設定（加えて、[Sweeps のハイパーパラメータチューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) を活用できるメリットがあります）

| Experiment   | Model Name       | Notes                                            | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | Overfit way too much on training data            | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | Current best model                               | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | Did worse than the baseline model. Need to debug | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | NOTES                                            | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B は、実験の CSV を取り込み、W&B Experiment Run に変換できます。以下の コードスニペット と スクリプト は、実験の CSV を取り込んでログする方法を示します。

1. まず CSV ファイルを読み込み、Pandas DataFrame に変換します。`"experiments.csv"` をあなたの CSV ファイル名に置き換えてください:

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

# 作業しやすいように Pandas DataFrame を整形する
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

2. 次に、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) で新しい W&B Run を開始し、W&B にトラッキングおよびログを行います:

    ```python
    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )
    ```

実験の実行中、すべてのメトリクスのインスタンスをログしておくと、W&B 上で閲覧・クエリ・分析できて便利です。これには [`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) コマンドを使用します:

```python
run.log({key: val})
```

[`define_metric`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) API を使って、Run の結果を表す最終的なサマリーメトリクスを任意でログできます。次の例では、`run.summary.update()` でサマリーメトリクスを Run に追加しています:

```python
run.summary.update(summaries)
```

サマリーメトリクスの詳細は、[サマリーメトリクスをログする]({{< relref path="./log-summary.md" lang="ja" >}}) を参照してください。

以下は、上のサンプル表を [W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) に変換する完全なサンプルスクリプトです:

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