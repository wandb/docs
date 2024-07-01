---
description: MLFlowからデータをインポートし、W&Bに保存したデータをエクスポートまたは更新します。
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# データのインポート & エクスポート

<head>
  <title>W&Bへのデータインポート & エクスポート</title>
</head>

W&BのパブリックAPIを使用して、データをエクスポートしたり、MLFlowからデータをインポートしたり、W&Bインスタンス間でデータをやり取りできます。

:::info
この機能にはpython>=3.8が必要です。
:::

## MLFlowからデータをインポート

W&Bは、Experiments、Runs、Artifacts、メトリクス、その他のメタデータを含むMLFlowからのデータインポートをサポートしています。

依存関係をインストール:

```shell
# 注: この操作にはpy38+が必要です
pip install wandb[importers]
```

W&Bにログインします。まだログインしていない場合は、プロンプトに従ってログインしてください。

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

デフォルトでは、`importer.collect_runs()` はMLFlowサーバーからすべてのRunsを収集します。特定のサブセットをアップロードしたい場合は、自分でRunsを作成してインポーターに渡すことができます。

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
DatabricksのMLFlowからインポートする場合は、最初に[Databricks CLIを設定](https://docs.databricks.com/dev-tools/cli/index.html)する必要があります。

先ほどのステップで、`mlflow-tracking-uri="databricks"` を設定してください。
:::

アーティファクトのインポートをスキップする場合は、`artifacts=False` を渡すことができます:

```py
importer.import_runs(runs, artifacts=False)
```

特定のW&Bエンティティとプロジェクトにインポートする場合は、`Namespace` を渡すことができます:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 別のW&Bインスタンスからデータをインポート

:::info
この機能はベータ版であり、W&Bパブリッククラウドからのインポートのみサポートしています。
:::

依存関係をインストール:

```sh
# 注: この操作にはpy38+が必要です
pip install wandb[importers]
```

ソースW&Bサーバーにログインします。まだログインしていない場合は、プロンプトに従ってログインしてください。

```sh
wandb login
```

ソースW&BインスタンスからデスティネーションW&Bインスタンスにすべてのrunsとartifactsをインポートします。runsとartifactsはデスティネーションインスタンスのそれぞれのネームスペースにインポートされます。

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# "entity/project" のすべてのruns, artifacts, reportsを
# srcからdstにインポートします
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... ここにさらにネームスペースを追加します
])
```

デスティネーションネームスペースを変更したい場合は、`remapping: dict[Namespace, Namespace]` を指定できます

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

デフォルトでは、インポートはインクリメンタルです。後続のインポートは以前の作業を検証しようとし、成功/失敗を記録する `.jsonl` ファイルに書き込みます。インポートが成功した場合、将来の検証はスキップされます。インポートが失敗した場合、それは再試行されます。これを無効にするには、`incremental=False` を設定します。

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### 既知の問題と制限

- デスティネーションネームスペースが存在しない場合、W&Bが自動的に作成します。
- デスティネーションネームスペースに同じIDのrunやartifactが存在する場合、W&Bはそれをインクリメンタルインポートとして扱います。デスティネーショ ンのrun/artifactは検証され、以前のインポートで失敗した場合は再試行されます。
- ソースシステムからのデータは削除されません。

1. バルクインポート中（特に大きなアーティファクトの場合）、S3のレート制限に達することがあります。 `botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation` というエラーメッセージが表示された場合、少数のネームスペースを移動させることでインポートの間隔をあけることができます。
2. Imported run tables appear to be blank in the workspace, but if you nav to the Artifacts tab and click the equivalent run table artifact you should see the table as expected.
3. システムメトリクスとカスタムチャート（`wandb.log`で明示的にログされていないもの）はインポートされません。

## データのエクスポート

W&Bに保存したデータをエクスポートまたは更新するには、パブリックAPIを使用します。このAPIを使用する前に、スクリプトからデータをログインする必要があります。詳細は[クイックスタート](../../quickstart.md)を参照してください。

**パブリックAPIのユースケース**

- **データのエクスポート**: Jupyter Notebookでカスタム分析のためにデータフレームを取得します。データを探索した後、新しい分析runを作成して結果をログインすることで、学びを同期できます。例: `wandb.init(job_type="analysis")`
- **既存のRunsの更新**: 特定のrunに対応するデータを更新できます。例えば、トレーニングrunのconfigに、最初にログされていなかったアーキテクチャやハイパーパラメータなどの追加情報を含めることができます。

利用可能な関数の詳細は、[Generated Reference Docs](../../ref/python/public-api/README.md) を参照してください。

### 認証

マシンを[APIキー](https://wandb.ai/authorize)で認証する方法は2つあります:

1. コマンドラインで `wandb login` を実行し、APIキーを貼り付けます。
2. `WANDB_API_KEY` 環境変数にAPIキーを設定します。

### runパスを見つける

パブリックAPIを使用するには、runパスが必要です。これは `<entity>/<project>/<run_id>` の形式です。アプリ UI で run のページを開き、[Overviewタブ](../app/pages/run-page.md#overview-tab)をクリックしてrunパスを取得します。

### runデータのエクスポート

完了したrunまたはアクティブなrunからデータをダウンロードします。一般的な使い方には、カスタム分析のためにJupyter notebookでデータフレームをダウンロードしたり、自動化された環境でカスタムロジックを使用することが含まれます。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

runオブジェクトの最も一般的に使用される属性は次のとおりです:

| 属性            | 意味                                                                                                                                                                                                                                                                                                                |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | runの設定情報を含む辞書です。例えば、トレーニングrunのハイパーパラメータやデータセットArtifactを作成するrunの前処理メソッドなどです。これらをrunの "入力" と考えてください。                                                                                                             |
| `run.history()` | モデルのトレーニング中に変化する値（例えば損失値）を格納する辞書のリストです。`wandb.log()` コマンドはこのオブジェクトに追加されます。                                                                                                                                                                                 |
| `run.summary`   | runの結果を要約する情報の辞書です。これには精度や損失などのスカラーや、大きなファイルが含まれます。デフォルトでは、`wandb.log()` は、ログされた時系列データの最終値をsummaryに設定します。summaryの内容は直接設定することもできます。summaryをrunの "出力" と考えてください。                                 |

過去のrunのデータを変更または更新することもできます。デフォルトでは、APIオブジェクトの単一インスタンスはすべてのネットワークリクエストをキャッシュします。実行中のスクリプトでリアルタイム情報を必要とする場合は、`api.flush()`を呼び出して更新された値を取得します。

### 異なる属性の理解

以下のrunについて

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

上記のrunオブジェクト属性に対する異なる出力は次の通りです

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

デフォルトのhistoryメソッドは、メトリクスを固定数のサンプルにサンプリングします（デフォルトは500で、`samples` \_\_引数でこれを変更できます）。大規模なrunのすべてのデータをエクスポートする場合は、`run.scan_history()` メソッドを使用できます。詳細は[APIリファレンス](https://docs.wandb.ai/ref/python/public-api) を参照してください。

### 複数のRunsのクエリ

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: 'Dataframes and CSVs', value: 'dataframes_csvs'},
{label: 'MongoDB Style', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

このスクリプト例では、プロジェクトを見つけ、名前、設定、要約統計を持つrunのCSVを出力します。`<entity>` と `<project>` をそれぞれW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summaryには、精度などのメトリクスの出力キー/値が含まれます。
    # ._json_dictを呼び出して大きなファイルを省略します
    summary_list.append(run.summary._json_dict)

    # .configにはハイパーパラメータが含まれます。
    # 特殊な値（_で始まるもの）を削除します
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameは人間が読みやすいrunの名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

  </TabItem>
  <TabItem value="mongoDB">

W&B APIは、`api.runs()` を使用してプロジェクト内の複数のrunsに対してクエリを実行する方法も提供します。一般的なユースケースは、カスタム分析のためにrunデータをエクスポートすることです。クエリインターフェースは[MongoDBが使用するもの](https://docs.mongodb.com/manual/reference/operator/query) と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs` を呼び出すと、リストのように動作する `Runs` オブジェクトが返されます。デフォルトでは、このオブジェクトは一度に50runsを順次読み込みますが、`per_page` キーワード引数を使用して読み込むページごとの数を変更できます。

`api.runs` は `order` キーワード引数も受け付けます。デフォルトの順序は `-created_at` で、昇順の `+created_at` を指定できます。`summary.val_acc` や `config.experiment_name` などの設定や要約値でのソートも可能です。

### エラーハンドリング

W&Bサーバーと通信中にエラーが発生した場合、 `wandb.CommError` が発生します。元の例外は `exc` 属性を介して調査できます。

### API を使用して最新のgitコミットを取得

UIでは、runをクリックしてから、runページのOverviewタブをクリックして最新のgitコミットを確認できます。また、`wandb-metadata.json` ファイルにも記載されています。パブリックAPIを使用してgitハッシュを取得するには、`run.commit`を使用します。

## よくある質問

### データをmatplotlibやseabornで可視化するためにエクスポートするにはどうすればよいですか？

一般的なエクスポートパターンについては、[APIの例](../../ref/python/public-api/README.md) をご覧ください。また、カスタムプロットや拡大されたrunsテーブルのダウンロードボタンをクリックすると、ブラウザからCSVをダウンロードできます。

### run中にrunの名前とIDを取得するにはどうすればよいですか？

`wandb.init()` を呼び出した後、スクリプトからランダムrun IDまたは人間が読みやすいrun名にアクセスできます。

- 一意のrun ID（8文字のハッシュ）: `wandb.run.id`
- ランダムrun名（人間が読みやすい）: `wandb.run.name`

runの識別子を設定する際に役立つ推奨事項は以下の通りです:

- **Run ID**: 生成されたハッシュのままにしておいてください。これはプロジェクト内のrun間で一意である必要があります。
- **Run name**: 読みやすく、できれば一意の短い名前を設定してください。これにより、チャート上の異なるラインを区別しやすくなります。
- **Run notes**: 実行中の内容を簡単に説明するのに最適な場所です。`wandb.init(notes="your notes here")` で設定できます。
- **Run tags**: runタグに動的にトラッキングし、UIのフィルターを使用して関心のあるrunsに絞り込むことができます。スクリプトからタグを設定し、UIで編集することができます。詳細な指示は[こちら](../app/features/tags.md) を参照してください。

## パブリックAPIの例

### runからメトリクスを読み取る

この例では、`wandb.log({"accuracy": acc})` で保存されたタイムスタンプと精度を出力します。保存先は `"<entity>/<project>/<run_id>"` です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Runsのフィルタリング

MongoDBクエリ言語を使用してフィルタリングできます。

#### 日付

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### runから特定のメトリクスを読み取る

runから特定のメトリクスを抽出するには、`keys` 引数を使用します。`run.history()` のデフォルトのサンプル数は500です。特定のメトリクスを含まないログステップは、出力データフレームに `NaN` として表示されます。`keys` 引数を使うと、指定されたメトリクスキーを含むステップをより頻繁にサンプルします。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2つのRunsを比較する

このスクリプトは、`run1` と `run2` の間で異なる設定パラメータを出力します。

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>, <project>, <run_id>を置き換えます
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")

df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

出力例:

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### run終了後にメトリクスを更新する

このスクリプト例では、以前のrunの精度を `0.9` に設定します。また、以前のrunの精度ヒストグラムを `numpy_array` のヒストグラムに変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### run終了後にメトリクス名を変更する

このスクリプト例では、要約テーブルのカラム名を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

:::caution
カラム名の変更はテーブルにのみ適用されます。チャートは元のメトリクス名を使用します。
:::

### 既存runの設定を更新する

このスクリプト例では、設定の1つを更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース消費をCSVファイルにエクスポート

以下のスクリプト例では、システムリソース消費を見つけて、それをCSVに保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サンプリングされていないメトリクスデータを取得

デフォルトでは、履歴からデータを取得する場合、それは500ポイントにサンプリングされます。`run.scan_history()` を使用してログされたすべてのデータポイントを取得します。以下は、履歴に記録されたすべての `loss` データポイントをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### 履歴からページネーションされたデータを取得

メトリクスがバックエンドで遅く取得される場合やAPIリクエストがタイムアウトする場合は、ページサイズを減らしてみてください。`scan_history` でページサイズを調整できます。デフォルトページサイズは500ですので、別のサイズを試して最適なものを探してください。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内のすべてのRunsからのメトリクスをCSVファイルにエクスポート

このスクリプトは、プロジェクト内のrunを取得し、それらの名前、設定、および要約統計を含むデータフレームとCSVを生成します。`<entity>` と `<project>` をそれぞれW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summaryにはメトリクスの出力キー/値が含まれます
    # 大きなファイルを省略するため ._json_dictを呼び出します
    summary_list.append(run.summary._json_dict)

    # .configにはハイパーパラメータが含まれます
    # 特殊な値（_で始まるもの）を削除します
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameは人間が読みやすいrunの名前です
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### runの開始時間を取得する

このスニペットコードは、runが作成された時間を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了したrunにファイルをアップロード

以下のスニペットコードは、指定されたファイルを完了したrunにアップロードします。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### runからファイルをダウンロード

以下のスニペットコードは、cifarプロジェクト内のrun ID uxte44z7に関連付けられた "model-best.h5" ファイルを見つけ、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### runからすべてのファイルをダウンロード

スニペットコードは、runに関連付けられたすべてのファイルを見つけ、それらをローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定のSweepからRunsを取得

このスニペットコードは、特定のSweepに関連するすべてのrunをダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweepから最も優れたrunを取得

以下のスニペットコードは、指定されたSweepから最も優れたrunを取得します。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は、Sweep設定で `metric` パラメータによって定義された最も優れたメトリクスを持つrunです。

### Sweepから最も優れたモデルファイルをダウンロード

このスニペットコードは、`model.h5` にモデルファイルを保存したSweepから、最も高い検証精度を持つモデルファイルをダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"Best run {runs[0].name} with {val_acc}% val accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")
```

### runから特定の拡張子を持つすべてのファイルを削除する

このスニペットコードは、runから指定された拡張子を持つすべてのファイルを削除します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

extension = ".png"
files = run.files()
for file in files:
    if file.name.endswith(extension):
        file.delete()
```

### システムメトリクスデータをダウンロード

このスニペットコードは、runのすべてのシステムリソース消費メトリクスを含むデータフレームを生成し、それをCSVに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 要約メトリクスを更新する

要約メトリクスを更新するには、辞書を渡すことができます。

```python
summary.update({"key": val})
```

### runを実行したコマンドを取得する

各runは、run概要ページでそれを開始したコマンドをキャプチャします。APIからこのコマンドを取得するには、以下を実行します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

