---
description: MLFlowからデータをインポートし、W&Bに保存したデータをエクスポートまたは更新する
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Import & Export Data

<head>  
  <title>データのインポート & エクスポート to W&B</title>
</head>

W&BのパブリックAPIを使用して、データをエクスポートしたり、MLFlowやW&Bインスタンス間でデータをインポートしたりすることができます。

:::info
この機能はpython>=3.8が必要です
:::

## MLFlowからデータをインポートする

W&Bは、MLFlowからデータをインポートすることをサポートしています。実験、Runs、Artifacts、メトリクス、その他のメタデータを含みます。

依存関係のインストール:

```shell
# 注意: これはpy38+が必要です
pip install wandb[importers]
```

W&Bにログインします。まだログインしていない場合はプロンプトに従ってください。

```shell
wandb login
```

既存のMLFlowサーバーからすべてのRunsをインポートします:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` はMLFlowサーバーからすべてのRunsを収集します。特定のサブセットをアップロードしたい場合は、独自のRunsのイテラブルを作成してインポーターに渡すことができます。

```py
import mlflow
from wandb.apis.importers.mlflow import MlflowRun

client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

runs: Iterable[MlflowRun] = []
for run in mlflow_client.search_runs(...):
    runs.append(MlflowRun(run, client))

importer.import_runs(runs)
```

:::tip
Databricks MLFlowからインポートする場合、最初に [Databricks CLIの設定](https://docs.databricks.com/dev-tools/cli/index.html) が必要になるかもしれません。

先のステップで`mlflow-tracking-uri="databricks"`を設定します。
:::

Artifactsのインポートをスキップするには、`artifacts=False`を渡します:

```py
importer.import_runs(runs, artifacts=False)
```

特定のW&Bエンティティとプロジェクトにインポートする場合は、`Namespace`を渡すことができます:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 別のW&Bインスタンスからデータをインポートする

:::info
この機能はベータ版であり、W&Bのパブリッククラウドからのインポートのみをサポートしています。
:::

依存関係のインストール:

```sh
# 注意: これはpy38+が必要です
pip install wandb[importers]
```

ソースW&Bサーバーにログインします。まだログインしていない場合はプロンプトに従ってください。

```sh
wandb login
```

ソースW&BインスタンスからデスティネーションW&BインスタンスにすべてのRunsとArtifactsをインポートします。RunsとArtifactsはデスティネーションインスタンスのそれぞれのNamespaceにインポートされます。

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# srcの"entity/project"からdstの"entity/project"へすべてのRuns、Artifacts、Reportsをインポートします
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... ここにさらにNamespaceを追加します
])
```

デスティネーションNamespaceを変更したい場合は、`remapping: dict[Namespace, Namespace]`を指定できます。

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

デフォルトでは、インポートはインクリメンタルです。次回のインポートでは前回の作業を検証し、成功/失敗を追跡する`.jsonl`ファイルに書き込もうとします。インポートが成功した場合、将来の検証はスキップされます。インポートが失敗した場合、再試行されます。これを無効にするには、`incremental=False`を設定します。

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### 既知の問題と制限事項

- デスティネーションNamespaceが存在しない場合、W&Bは自動的に1つ作成します。
- デスティネーションNamespaceに同じIDを持つRunまたはArtifactがある場合、W&Bはこれをインクリメンタルインポートとして扱います。デスティネーションのRun/Artifactは検証され、前回のインポートで失敗した場合は再試行されます。
- ソースシステムからデータが削除されることはありません。

1. バルクインポート（特に大きいArtifactsのインポート）時にS3のレート制限に引っかかることがあります。`botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation`が表示された場合、インポートを間隔を空けて行い、少数のNamespaceを一度に移動させることを試してみてください。
2. インポートされたRunテーブルがWorkspaceで空白に見えることがありますが、Artifactsタブに移動して同等のRunテーブルArtifactをクリックすると期待通りにテーブルが表示されます。
3. システムメトリクスとカスタムチャート（`wandb.log`で明示的にログされていないもの）はインポートされません。

## データのエクスポート

パブリックAPIを使用して、W&Bに保存したデータをエクスポートまたは更新します。このAPIを使用する前に、スクリプトからデータをログする必要があります。詳細は[クイックスタート](../../quickstart.md)を参照してください。

**パブリックAPIのユースケース**

- **データのエクスポート**: Jupyter Notebookでカスタム分析を行うためにデータフレームをダウンロードします。データを探索した後、新しい分析Runを作成し、結果をログしてあなたの学びを同期することができます（例：`wandb.init(job_type="analysis")`）。
- **既存のRunsの更新**: W&BのRunと関連付けられたデータを更新することができます。例えば、以前ログしなかった追加情報（アーキテクチャーやハイパーパラメーターなど）を含めるために、Runsの設定を更新することができます。

利用可能な関数の詳細については、[生成されたリファレンスドキュメント](../../ref/python/public-api/README.md)を参照してください。

### 認証

以下の2つの方法のいずれかで、[APIキー](https://wandb.ai/authorize)を使用してマシンを認証します:

1. コマンドラインで`wandb login`を実行し、APIキーを貼り付けます。
2. `WANDB_API_KEY`環境変数にAPIキーを設定します。

### Runパスの取得

パブリックAPIを使用するには、しばしばRunパス（ `<entity>/<project>/<run_id>`）が必要です。アプリのUIでRunページを開き、[Overviewタブ](../app/pages/run-page.md#overview-tab)をクリックしてRunパスを取得します。

### Runデータのエクスポート

終了しているまたはアクティブなRunからデータをダウンロードします。一般的な使用例には、Jupyterノートブックでのカスタム分析のためのデータフレームダウンロードや、カスタムロジックを用いた自動化環境での使用が含まれます。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

Runオブジェクトで最も一般的に使用される属性は以下の通りです:

| Attribute       | 意味                                                                                                                                                                                                                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | Runの設定情報の辞書。トレーニングRunのハイパーパラメーターやデータセットArtifactを作成するRunの前処理メソッドなどが含まれます。これらをRunの「入力」と考えます。                                                                                                    |
| `run.history()` | モデルがトレーニング中に変化する値を保存するための辞書のリスト。`wandb.log()` コマンドがこのオブジェクトに追加されます。                                                                                                                                                                 |
| `run.summary`   | Runの結果を要約する情報の辞書。精度や損失のようなスカラー値や大きなファイルが含まれます。デフォルトでは、`wandb.log()` がログされた時系列の最終値をsummaryに設定します。summaryの内容は直接設定することもできます。summaryをRunの「出力」と考えます。 |

また、過去のRunsのデータを修正または更新することもできます。デフォルトでは、1つのAPIオブジェクトインスタンスはすべてのネットワーク要求をキャッシュします。実行中のスクリプトでリアルタイム情報が必要な場合は、`api.flush()` を呼び出して更新された値を取得します。

### 属性の理解

以下のRunの例

```python
n_epochs = 5
config = {"n_epochs": n_epochs}
run = wandb.init(project=project, config=config)
for n in range(run.config.get("n_epochs")):
    run.log(
        {"val": random.randint(0, 1000), "loss": (random.randint(0, 1000) / 1000.00)}
    )
run.finish()
```

このRunオブジェクト属性の異なる出力結果です。

#### `run.config`

```python
{"n_epochs": 5}
```

#### `run.history()`

```shell
   _step  val   loss  _runtime  _timestamp
0      0  500  0.244         4  1644345412
1      1   45  0.521         4  1644345412
2      2  240  0.785         4  1644345412
3      3   31  0.305         4  1644345412
4      4  525  0.041         4  1644345412
```

#### `run.summary`

```python
{
    "_runtime": 4,
    "_step": 4,
    "_timestamp": 1644345412,
    "_wandb": {"runtime": 3},
    "loss": 0.041,
    "val": 525,
}
```

### サンプリング

デフォルトのhistoryメソッドはメトリクスを固定数のサンプルにサンプリングします（デフォルトは500、`samples`引数で変更可能）。大規模なRunのすべてのデータをエクスポートしたい場合は、`run.scan_history()`メソッドを使用できます。詳細については[APIリファレンス](https://docs.wandb.ai/ref/python/public-api)を参照してください。

### 複数のRunsのクエリ

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: 'データフレームとCSV', value: 'dataframes_csvs'},
{label: 'MongoDBスタイル', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

このサンプルスクリプトは、プロジェクトを見つけてRunの名前、設定、およびsummaryスタッツのCSVを出力します。`<entity>`と`<project>`をW&Bエンティティとプロジェクトの名前で置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summaryには精度などの
    # メトリクス用の出力キー/値が含まれます。
    # ._json_dictを呼び出して大きなファイルを省略します
    summary_list.append(run.summary._json_dict)

    # .configにはハイパーパラメータが含まれます。
    # _で始まる特殊な値を削除します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameはRunの人間が読める名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

  </TabItem>
  <TabItem value="mongoDB">

W&B APIはまた、api.runs()を使ってプロジェクト内のRunsを横断してクエリを行う方法も提供しています。最も一般的なユースケースはカスタム分析のためにRunsデータをエクスポートすることです。クエリインターフェースは[MongoDB](https://docs.mongodb.com/manual/reference/operator/query)と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs`を呼び出すと、`Runs`オブジェクトが返されます。これは反復可能でリストのように動作します。デフォルトでは、オブジェクトは必要に応じてシーケンスで50のRunsを一度にロードしますが、`per_page`キーワード引数でロードするページごとの数を変更できます。

`api.runs`はまた、`order`キーワード引数を受け付けます。デフォルトの順序は`-created_at`です。昇順で結果を取得するには`+created_at`を指定します。configまたはsummaryの値でもソートできます。例: `summary.val_acc`または`config.experiment_name`。

### エラーハンドリング

W&Bサーバーとの通信中にエラーが発生した場合、`wandb.CommError`が発生します。元の例外は`exc`属性を通じて調査できます。

### APIを通じて最新のgitコミットを取得する

UIでは、Runをクリックした後、RunページのOverviewタブをクリックして最新のgitコミットを表示します。また、`wandb-metadata.json`ファイルにもあります。パブリックAPIを使用して、`run.commit`でgitハッシュを取得できます。

## よくある質問

### matplotlibやseabornで可視化するためにデータをエクスポートするにはどうすればいいですか？

一般的なエクスポートパターンについては、[APIの例](../../ref/python/public-api/README.md)をチェックしてください。カスタムプロットや展開したRunテーブルのダウンロードボタンをクリックして、ブラウザからCSVをダウンロードすることもできます。

### Runの実行中にRunの名前やIDを取得するには？

`wandb.init()`を呼び出した後、スクリプトからランダムRun IDや人間が読めるRunの名前にアクセスできます:

- ユニークRun ID（8文字のハッシュ）: `wandb.run.id`
- ランダムRunの名前（人間が読める）: `wandb.run.name`

有用なRun識別子を設定する方法を考えている場合、次のものをお勧めします:

- **Run ID**: 生成されたハッシュのままにしておきます。プロジェクト内のRuns間で一意である必要があります。
- **Runの名前**: 短く、読みやすく、可能であれば一意にして、チャート上の異なるラインを区別できるようにします。
- **Runノート**: Runで何をしているのかを簡単に説明するのに適しています。`wandb.init(notes="your notes here")`で設定できます。
- **Runタグ**: Runタグで動的にトラックし、UIのフィルターを使用して自分が気にしているRunsにテーブルを絞り込みます。スクリプトからタグを設定し、UIでも編集できます。詳細な手順は[ここ](../app/features/tags.md)を参照してください。

## 公共APIの例

### Runからメトリクスを読み取る

この例では、`wandb.log({"accuracy": acc})`で保存されたRunのタイムスタンプと精度を出力します。対象のRunは`"<entity>/<project>/<run_id>"`に保存されています。

```python
import wandb

api = wandb.Api()

