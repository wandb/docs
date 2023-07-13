---
description: Importing and logging data into W&B
displayed_sidebar: ja
---

# CSVファイルからデータと実験をログに記録する

<head>
  <title>W&Bの実験でCSVファイルをトラッキング</title>
</head>

W&B Pythonライブラリを使ってCSVファイルをログに記録し、[W&Bダッシュボード](../app.md)で可視化します。W&Bダッシュボードは、機械学習モデルの結果を整理し、可視化するための中心的な場所です。これは、W&Bに記録されていない[以前の機械学習実験の情報が含まれているCSVファイル](#import-and-log-your-csv-of-experiments)や、[データセットが含まれているCSVファイル](#import-and-log-your-dataset-csv-file)がある場合に特に便利です。

## データセットのCSVファイルをインポートして記録する

<!-- {% embed url="https://drive.google.com/file/d/1jBG3M4VnaMgeclRzowYZEYvFxvwb9SXF/view?usp=sharing" %} -->

CSVファイルの内容を再利用しやすくするために、W&Bアーティファクトを利用することをお勧めします。

1. まず、CSVファイルをインポートします。次のコードスニペットで、`iris.csv`ファイル名を自分のCSVファイル名に置き換えてください。

```python
import wandb
import pandas as pd

# 新しいDataFrameにCSVを読み込む
new_iris_dataframe = pd.read_csv("iris.csv")
```

2. CSVファイルをW&Bテーブルに変換して、[W&Bダッシュボード](../app.md)を利用します。

```python
# DataFrameをW&Bテーブルに変換
iris_table = wandb.Table(dataframe=new_iris_dataframe)
```

3. 次に、W&B Artifactを作成し、テーブルをArtifactに追加します：

```python
# 行の上限を200000に増やし、再利用しやすくするために、
# テーブルをArtifactに追加
iris_table_artifact = wandb.Artifact(
    "iris_artifact", 
    type="dataset"
    )        
iris_table_artifact.add(iris_table, "iris_table")

# 生のcsvファイルをArtifact内にログして、データを保持する
iris_table_artifact.add_file("iris.csv")
```
W&B Artifactsに関する詳細は、[Artifactsの章](../../artifacts/intro.md)を参照してください。

4. 最後に、`wandb.init`を使って新しいW&B Runを開始し、W&Bにトラッキングとログを行います：

```python
# データをログするW&B runを開始する
run = wandb.init(project="Tables-Quickstart")

# テーブルをログして可視化するためのrunに追加...
run.log({"iris": iris_table})

# そして、利用可能な行制限を増やすためにArtifactとしてログ！
run.log_artifact(iris_table_artifact)
```
`wandb.init()` APIは、データをRunにログするために新しいバックグラウンドプロセスを生成し、デフォルトでwandb.aiにデータを同期します。W&Bワークスペースダッシュボードでリアルタイムの可視化が表示されます。以下の画像は、コードスニペットのデモンストレーションの出力を示しています。

![CSVファイルがW&Bダッシュボードにインポートされた状態](/images/track/import_csv_tutorial.png)

上記のコードスニペットを含む完全なスクリプトは以下になります:

```python
import wandb
import pandas as pd

# CSVを新しいDataFrameに読み込む
new_iris_dataframe = pd.read_csv("iris.csv")

# DataFrameをW&Bテーブルに変換する
iris_table = wandb.Table(dataframe=new_iris_dataframe)

# テーブルをアーティファクトに追加して、行の制限を200000に増やし、再利用しやすくする
iris_table_artifact = wandb.Artifact(
    "iris_artifact", 
    type="dataset"
    )        
iris_table_artifact.add(iris_table, "iris_table")

# 生のcsvファイルをアーティファクトの中にログして、データを保護する
iris_table_artifact.add_file("iris.csv")

# W&Bのrunを開始してデータをログする
run = wandb.init(project="Tables-Quickstart")
# ランでテーブルを可視化するためにログに記録する...
run.log({"iris": iris_table})

# そして、利用可能な行制限を増やすためにアーティファクトとしてログに記録する！
run.log_artifact(iris_table_artifact)

# ランを終了する（ノートブックで便利）
run.finish()
```

## CSV形式の実験名をインポートしてログに記録する

<!-- {% embed url="https://drive.google.com/file/d/1PL4RSdopHEptDR5Gi0DEzECXuoW_5B0f/view?usp=sharing" %}
以下の表は、変換後にこのWeights & Biasesダッシュボードになります
{% endembed %} -->

場合によっては、実験の詳細がCSVファイルに記録されていることがあります。このようなCSVファイルに一般的に含まれる詳細には以下のものがあります。

* 実験ランの名前
* 最初の[ノート](../../app/features/notes.md)
* 実験を識別するための[タグ](../../app/features/tags.md)
* 実験に必要な構成（[Sweeps Hyperparameter Tuning](../../sweeps/intro.md)を活用できる利点付き）。

| 実験           | モデル名              | ノート                                             | タグ             | レイヤー数 | 最終トレーニング精度 | 最終検証精度 | トレーニング損失                          |
| ------------ | ---------------- | ------------------------------------------------ | ------------- | ---------- | --------------- | ------------- | ------------------------------------- |
| 実験 1          | mnist-300-レイヤー | トレーニングデータで過学習が起こった                | \[最新]            | 300        | 0.99            | 0.90          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 実験 2          | mnist-250-レイヤー | 現在の最優れたモデル                                   | \[プロダクション, 最高] | 250        | 0.95            | 0.96          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| 実験 3          | mnist-200-レイヤー | ベースラインモデルよりも悪い結果。デバッグが必要         | \[デバッグ]         | 200        | 0.76            | 0.70          | \[0.55, 0.45, 0.44, 0.42, 0.40, 0.39] |
| ...          | ...              | ...                                              | ...           | ...        | ...             | ...           |                                       |
| 実験 N          | mnist-X-レイヤー   | ノート                                             | ...           | ...        | ...             | ...           | \[..., ...]                           |
W&Bは、実験のCSVファイルをW&B Experiment Runに変換することができます。以下のコードスニペットとコードスクリプトは、実験のCSVファイルをインポートしてログに記録する方法を示しています:

1. まず、CSVファイルを読み込んでPandas DataFrameに変換します。`"experiments.csv"`をあなたのCSVファイルの名前に置き換えてください。

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

# Pandas DataFrameを扱いやすくするために書式を整える
for i, row in loaded_experiment_df.iterrows():
    
    run_name = row[EXPERIMENT_NAME_COL]
    notes = row[NOTES_COL]
    tags = row[TAGS_COL]

    config = {}
    for config_col in CONFIG_COLS:
        config[config_col] = row[config_col]
メトリクス = {}
    for metric_col in METRIC_COLS:
        metrics[metric_col] = row[metric_col]
    
    summaries = {}
    for summary_col in SUMMARY_COLS:
        summaries[summary_col] = row[summary_col]
```

2. 次に、新しいW&B Runを開始して、[`wandb.init()`](../../../ref/python/init.md)を使用してW&Bにトラッキングとログを行います。

```python
run = wandb.init(
    project=PROJECT_NAME,
    name=run_name,
    tags=tags,
    notes=notes,
    config=config
    )    
```

実験が実行されている間、メトリクスのすべてのインスタンスをログして、W&Bで表示、クエリ、分析できるようにすることができます。これを実現するためには、[`run.log()`](../../../ref/python/log.md)コマンドを使用します。

```python
run.log({key: val})
```

オプションで、runの結果を定義するための最終的なサマリーメトリクスをログに残すことができます。これを実現するためには、W&B [`define_metric`](../../../ref/python/run.md#define_metric) APIを使用します。この例では、`run.summary.update()`を使って、サマリーメトリクスをrunに追加します。
```python
run.summary.update(summaries)
```

詳細なサマリーメトリクスについては、[ログサマリーメトリクス](./log-summary.md)を参照してください。

以下は、上記のサンプルテーブルを[W&B ダッシュボード](../app.md) に変換する完全な例スクリプトです。

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
```
メトリクス = {}

    for metric_col in METRIC_COLS:

        metrics[metric_col] = row[metric_col]



    サマリーズ = {}

    for summary_col in SUMMARY_COLS:

        summaries[summary_col] = row[summary_col]



    run = wandb.init(project=PROJECT_NAME, name=run_name, \

    tags=tags, notes=notes, config=config)



    for key, val in metrics.items():

        if isinstance(val, list):

            for _val in val:

                run.log({key: _val})

        else:

            run.log({key: val})

            

    run.summary.update(summaries)

    run.finish()

```