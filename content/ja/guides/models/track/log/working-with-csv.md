---
title: CSV ファイルを実験と一緒にトラッキングする
description: W&B へのデータのインポートとログ
menu:
  default:
    identifier: ja-guides-models-track-log-working-with-csv
    parent: log-objects-and-media
---

W&B Python ライブラリを使って CSV ファイルをログし、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で可視化しましょう。W&B ダッシュボードは、機械学習モデルから得られる結果を整理・可視化する中心的な場所です。これは、[過去の機械学習実験情報が記載された CSV ファイル]({{< relref path="#import-and-log-your-csv-of-experiments" lang="ja" >}}) を W&B にログしていない場合や、[データセットが含まれた CSV ファイル]({{< relref path="#import-and-log-your-dataset-csv-file" lang="ja" >}}) がある場合に特に役立ちます。

## データセットの CSV ファイルをインポートしてログする

W&B Artifacts を利用して、CSV ファイルの内容を再利用しやすくすることをおすすめします。

1. まず最初に、CSV ファイルをインポートしましょう。以下のコードスニペットでは、`iris.csv` をご自身の CSV ファイル名に置き換えてください。

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSV ファイルを W&B Table に変換し、[W&B ダッシュボード]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}) で活用します。

```python
# DataFrame を W&B Table に変換する
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifact を作成し、Table を Artifacts に追加します。

```python
# Table を Artifact に追加して
# 行数上限を 200000 に増やし再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 元の csv ファイルも artifact 内に保存してデータを保護
iris_table_artifact.add_file("iris.csv")
```
W&B Artifacts について、詳しくは [Artifacts チャプター]({{< relref path="/guides/core/artifacts/" lang="ja" >}}) をご覧ください。

4. 最後に、新しい W&B Run を開始し、`wandb.init` でログを記録・追跡しましょう。

```python
# W&B run を開始してデータをログ
run = wandb.init(project="tables-walkthrough")

# Table を run として可視化用にログする
run.log({"iris": iris_table})

# そして Artifact としてもログして行数上限を拡大!
run.log_artifact(iris_table_artifact)
```

`wandb.init()` API はバックグラウンドで新しいプロセスを立ち上げ、Run にデータをログします。そしてデータは（デフォルトで）wandb.ai に同期されます。W&B Workspace ダッシュボードでライブ可視化を確認できます。以下の画像は、上記コードスニペットの実行結果の例です。

{{< img src="/images/track/import_csv_tutorial.png" alt="CSV file imported into W&B Dashboard" >}}

この一連のコードスニペットをまとめた全体スクリプトは以下です。

```python
import wandb
import pandas as pd

# CSV を新しい DataFrame に読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrame を W&B Table に変換する
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# Table を Artifact に追加して
# 行数上限を 200000 に増やし再利用しやすくする
iris_table_artifact = wandb.Artifact("iris_artifact", type="dataset")
iris_table_artifact.add(iris_table, "iris_table")

# 元の csv ファイルも artifact 内に保存してデータを保護
iris_table_artifact.add_file("iris.csv")

# W&B run を開始してデータをログ
run = wandb.init(project="tables-walkthrough")

# Table を run として可視化用にログする
run.log({"iris": iris_table})

# そして Artifact としてもログして行数上限を拡大!
run.log_artifact(iris_table_artifact)

# run を終了（ノートブック利用時に便利）
run.finish()
```

## 実験の CSV ファイルをインポートしてログする

場合によっては、実験の詳細を CSV ファイルとして持っていることもあります。このような CSV ファイルによく含まれる情報は以下の通りです。

* 実験 run の名前
* 初期の [ノート]({{< relref path="/guides/models/track/runs/#add-a-note-to-a-run" lang="ja" >}})
* [タグ]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}}) で実験を区別
* 必要な設定（我々の [Sweeps ハイパーパラメータチューニング]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) も利用できます）

| Experiment   | Model Name       | Notes                                      | Tags          | Num Layers | Final Train Acc | Final Val Acc | Training Losses                       |
| ------------ | ---------------- | ------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| Experiment 1 | mnist-300-layers | トレーニングデータに過学習しすぎ            | \[latest]     | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 2 | mnist-250-layers | 現在のベストモデル                         | \[prod, best] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| Experiment 3 | mnist-200-layers | ベースラインよりパフォーマンスが悪い 要調査 | \[debug]      | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                        | ...           | ...        | ...             | ...           |                                       |
| Experiment N | mnist-X-layers   | ノート                                    | ...           | ...        | ...             | ...           | \[..., ...]                           |

W&B では、実験記録の入った CSV ファイルを読み込み、W&B Experiment Run へと変換できます。以下のコードスニペット・スクリプトは、実験記録用の CSV ファイルをインポート＆ログする方法を示します。

1. まずは、CSV ファイルを読み込んで Pandas DataFrame に変換します。`"experiments.csv"` をご自身のファイル名に置き換えてください。

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

# Pandas DataFrame を処理しやすく整形する
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

2. 次に、[`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init" lang="ja" >}}) を使って新しい W&B Run を開始し、W&B への記録・追跡をはじめます。

    ```python
    run = wandb.init(
        project=PROJECT_NAME, name=run_name, tags=tags, notes=notes, config=config
    )
    ```

実験を進める間、メトリクスごとに毎回ログを記録しておくと、W&B 上で可視化・分析できます。これを行うには、[`run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog" lang="ja" >}}) コマンドを使用します。

```python
run.log({key: val})
```

run の成果として最終的なサマリーメトリクスを記録する場合は、[`define_metric`]({{< relref path="/ref/python/sdk/classes/run#define_metric" lang="ja" >}}) API を使えます。この例では `run.summary.update()` でサマリーメトリクスを追加しています。

```python
run.summary.update(summaries)
```

サマリーメトリクスについて詳しくは、[Log Summary Metrics]({{< relref path="./log-summary.md" lang="ja" >}}) をご参照ください。

上記のサンプル表を W&B ダッシュボードに変換する全体スクリプト例は下記の通りです。

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