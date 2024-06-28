---
description: MLFlowからデータをインポートし、W&Bに保存したデータをエクスポートまたは更新する
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Import & Export Data

<head>
  <title>W&Bへのデータのインポートとエクスポート</title>
</head>

W&BのパブリックAPIを使って、データをエクスポートしたりMLFlowやW&B間でデータをインポートしたりすることができます。

:::info
この機能には python>=3.8 が必要です
:::

## MLFlowからデータをインポートする

W&Bは、MLFlowからのデータインポートをサポートしており、Experiments、Runs、Artifacts、メトリクス、その他のメタデータを含みます。

依存関係をインストールします：

```shell
# 注意: この操作には py38+ が必要です
pip install wandb[importers]
```

W&Bにログインします。以前にログインしたことがない場合はプロンプトに従ってください。

```shell
wandb login
```

既存のMLFlowサーバーからすべてのRunsをインポートします：

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` はMLFlowサーバーからすべてのRunsを収集します。特定のサブセットをアップロードしたい場合は、独自のRunsのイテレータブルを構築し、インポーターに渡すことができます。

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
もしDatabricks MLFlowからインポートする場合、先に [Databricks CLIの設定](https://docs.databricks.com/dev-tools/cli/index.html) をする必要があるかもしれません。

前述のステップで `mlflow-tracking-uri="databricks"` を設定してください。
:::

Artifactsのインポートをスキップしたい場合は、`artifacts=False` を渡すことができます：

```py
importer.import_runs(runs, artifacts=False)
```

特定のW&Bエンティティとプロジェクトにインポートするには、`Namespace` を渡すことができます：

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## 別のW&Bインスタンスからデータをインポートする

:::info
この機能はベータ版であり、W&Bのパブリッククラウドからのインポートのみをサポートしています。
:::

依存関係をインストールします：

```sh
# 注意: この操作には py38+ が必要です
pip install wandb[importers]
```

ソースW&Bサーバーにログインします。以前にログインしたことがない場合はプロンプトに従ってください。

```sh
wandb login
```

ソースW&Bインスタンスから目的地のW&BインスタンスにすべてのRunsとArtifactsをインポートします。RunsとArtifactsはそれぞれの名前空間にインポートされます。

```py
from wandb.apis.importers.wandb import WandbImporter
from wandb.apis.importers import Namespace

importer = WandbImporter(
    src_base_url="https://api.wandb.ai",
    src_api_key="your-api-key-here",
    dst_base_url="https://example-target.wandb.io",
    dst_api_key="target-environment-api-key-here",
)

# "entity/project" のRuns, Artifacts, Reportsをすべてsrcからdstにインポートします
importer.import_all(namespaces=[
    Namespace(entity, project),
    # ... ここにさらに名前空間を追加します
])
```

目的地の名前空間を変更したい場合、`remapping: dict[Namespace, Namespace]` を指定できます

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    remapping={
        Namespace(entity, project): Namespace(new_entity, new_project),
    }
)
```

デフォルトでは、インポートはインクリメンタルです。後続のインポートは前回の成果を検証し、成功/失敗を追跡する`.jsonl`ファイルに書き込みます。インポートが成功した場合、次回の検証はスキップされます。インポートが失敗した場合、再試行されます。これを無効にするには、`incremental=False` を設定します。

```py
importer.import_all(
    namespaces=[Namespace(entity, project)],
    incremental=False,
)
```

### 既知の問題と制限

- 目的地の名前空間が存在しない場合、W&Bはそれを自動的に作成します。
- 名前空間で同じIDを持つRunやArtifactがある場合、W&Bはそれをインクリメンタルなインポートと見なします。目的地のRun/Artifactが以前のインポートで失敗した場合は再試行されます。
- データは決してソースシステムから削除されません。

1. 大量のインポート（特に大きなArtifacts）の場合、S3のレート制限に遭遇することがあります。`botocore.exceptions.ClientError: An error occurred (SlowDown) when calling the PutObject operation` というエラーが表示された場合は、名前空間を少しずつ移動してインポートを分散させてみてください。
2. インポートされたRunテーブルがWorkspaceで空白に見えることがありますが、Artifactsタブに移動してRunテーブルのArtifactをクリックすれば、期待通りにテーブルが表示されます。
3. システムメトリクスとカスタムチャート（`wandb.log`で明示的にログされないもの）はインポートされません。

## データのエクスポート

パブリックAPIを使用して、W&Bに保存したデータをエクスポートしたり更新したりします。このAPIを使用する前に、スクリプトからデータをログする必要があります。詳細は[クイックスタート](../../quickstart.md) を確認してください。

**パブリックAPIのユースケース**

- **データのエクスポート**: カスタム分析のためにJupyter Notebookにデータフレームを取り込む。データを探索した後、新しい分析Runを作成して結果をログすることで、学びを同期できます。例: `wandb.init(job_type="analysis")`
- **既存のRunsの更新**: W&B Runに関連するログデータを更新できます。例えば、追加情報（アーキテクチャや元々ログされていなかったハイパーパラメータなど）を含めるために一連のRunsの設定を更新する場合です。

利用可能な関数の詳細は [生成されたリファレンスドキュメント](../../ref/python/public-api/README.md) を参照してください。

### 認証

次のいずれかの方法で、マシンを認証し、[APIキー](https://wandb.ai/authorize) を設定します：

1. `wandb login` をコマンドラインで実行し、APIキーを入力する。
2. `WANDB_API_KEY` 環境変数にAPIキーを設定する。

### Runパスの取得

パブリックAPIを使用するには、しばしばRunパス `<entity>/<project>/<run_id>` が必要です。アプリのUIでRunページを開き、[Overviewタブ](../app/pages/run-page.md#overview-tab)をクリックしてRunパスを取得します。

### Runデータのエクスポート

終了またはアクティブなRunからデータをダウンロードします。一般的な使用例としては、カスタム分析のためにデータフレームをダウンロードする場合や、カスタムロジックを自動化された環境で使用する場合などがあります。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

Runオブジェクトの最も一般的に使用される属性は次の通りです：

| 属性          | 意味                                                                                                                                                                                                                                                   |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`  | Runの設定情報の辞書であり、例えばトレーニングRunのハイパーパラメータやデータセットArtifactを作成するRunの前処理方法などが含まれます。これをRunの「入力」と考えてください。                                                                               |
| `run.history()` | モデルのトレーニング中に変化する値を保存するための辞書のリスト。コマンド`wandb.log()`はこのオブジェクトに追加します。                                                                                                                               |
| `run.summary` | Runの結果を要約する情報の辞書です。これには、精度や損失などのスカラー値や大規模なファイルが含まれます。デフォルトでは、`wandb.log()`はログされた時系列の最終値をサマリーに設定します。サマリーの内容は直接設定することもできます。サマリーをRunの「出力」と考えてください。|

過去のRunsのデータを修正または更新することもできます。デフォルトでは、1つのAPIオブジェクトインスタンスはすべてのネットワークリクエストをキャッシュします。ランニングスクリプトでリアルタイム情報が必要なユースケースの場合、`api.flush()`を呼び出して更新値を取得します。

### 異なる属性の理解

以下のRunの場合

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

上記のRunオブジェクト属性の異なる出力は次のとおりです。

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

デフォルトのhistoryメソッドは、固定数のサンプルにメトリクスをサンプリングします（デフォルトは500、`samples` 引数で変更可能）。大規模なRunのすべてのデータをエクスポートしたい場合、`run.scan_history()` メソッドを使用できます。詳細は [APIリファレンス](https://docs.wandb.ai/ref/python/public-api) を参照してください。

### 複数のRunsのクエリ

<Tabs
defaultValue="dataframes_csvs"
values={[
{label: 'Dataframes and CSVs', value: 'dataframes_csvs'},
{label: 'MongoDB Style', value: 'mongoDB'},
]}>
<TabItem value="dataframes_csvs">

この例のスクリプトはプロジェクトを見つけ、名前、設定、サマリースタッツを含むRunsのCSVを出力します。`<entity>` と `<project>` をそれぞれあなたのW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summaryには、精度などのメトリクスの出力キー/値が含まれています。
    #  大きなファイルを省略するために ._json_dict を呼び出します
    summary_list.append(run.summary._json_dict)

    # .configにはハイパーパラメータが含まれています。
    #  特殊な値（ _ で始まるもの）を取り除きます。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameはRunの人間が読み取れる名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

  </TabItem>
  <TabItem value="mongoDB">

W&B APIは、api.runs() を使ってプロジェクト内のRunsをクエリする方法も提供します。最も一般的なユースケースはカスタム分析のためのRunsデータのエクスポートです。クエリインターフェースは [MongoDBが使用するもの](https://docs.mongodb.com/manual/reference/operator/query) と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs` を呼び出すと、イテレータブルでリストのように動作する `Runs` オブジェクトが返されます。デフォルトでは、必要に応じて一度に50件のRunsが順次読み込まれますが、`per_page` というキーワード引数でロードする数を変更できます。

`api.runs` は `order` キーワード引数も受け付けます。デフォルトの順序は `-created_at` で、昇順で結果を取得するには `+created_at` を指定します。configやsummaryの値でソートすることもできます。例: `summary.val_acc` や `config.experiment_name`

### エラーハンドリング

W&Bサーバーと通信中にエラーが発生した場合、`wandb.CommError` が発生します。元の例外は `exc` 属性を通じて内省できます。

### APIを通じて最新のgitコミットを取得する

UIでは、RunをクリックしてからRunページのOverviewタブをクリックして最新のgitコミットを確認できます。また、`wandb-metadata.json`ファイルにもあります。パブリックAPIを使用して、`run.commit`でgitハッシュを取得できます。

## よくある質問

### matplotlibやseabornで可視化するためにデータをエクスポートするには？

一般的なエクスポートパターンについては、[APIの例](../../ref/python/public-api/README.md) をご覧ください。また、カスタムプロットや展開されたRunsテーブルでダウンロードボタンをクリックして、ブラウザからCSVをダウンロードすることもできます。

### Run中にRunの名前やIDを取得するには？

`wandb.init()`を呼び出した後、スクリプトからランダムなRun IDまたは人間が読み取れるRun名にアクセスできます：

- ユニークなRun ID（8文字のハッシュ）: `wandb.run.id`
- ランダムなRun名（人間が読み取れる）: `wandb.run.name`

有用な識別子をRunsに設定する方法について検討している場合は、次のことをお勧めします：

- **Run ID**: 生成されたハッシュのままにしておきます。これはプロジェクト内のRuns間でユニークである必要があります。
- **Run名**: これは短く、読みやすく、できればユニークであるべきです。これにより、チャート上の異なるラインを区別できます。
- **Runノーツ**: これは、Runで行っていることを簡単に記述するのに適しています。`wandb.init(notes="your notes here")` で設定できます。
- **Runタグ**: Runタグで動的に追跡し、UIのフィルターを使用して関心のあるRunsだけをフィルタリングします。スクリプトからタグを設定し、UIでRunsテーブルやRunページのOverviewタブで編集できます。詳しい手順は [こちら](../app/features/tags.md) を参照してください。

## パブリックAPIの例

### Runからメトリクスを読み取る

この例では、`wandb.log({"accuracy": acc})` で保存されたRunのタイムスタンプと精度を出力します。保存先は `"<entity>/<project>/<run_id>"` です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print

### 実行が終了した後の run メトリクスを更新する

この例では、以前の run の精度を `0.9` に設定しています。また、以前の run の精度ヒストグラムを `numpy_array` のヒストグラムに変更しています。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 実行が終了した後の run メトリクスをリネームする

この例では、テーブル内のサマリー列をリネームします。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

:::caution
列のリネームはテーブルにのみ適用されます。チャートは元の名前のメトリクスを参照し続けます。
:::

### 既存の run の設定を更新する

この例では、設定の一つを更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース消費量を CSV ファイルにエクスポートする

以下のスニペットは、システムリソース消費量を取得し、それを CSV に保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サンプリングされていないメトリクスデータを取得する

履歴からデータを取得すると、デフォルトでは 500 ポイントにサンプリングされます。`run.scan_history()` を使用して、ログに記録されたすべてのデータポイントを取得します。以下は、履歴にログされたすべての `loss` データポイントをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### 履歴からページネーションされたデータを取得する

バックエンドや API リクエストがタイムアウトする場合、`scan_history` のページサイズを下げることで個々のリクエストがタイムアウトしないようにできます。デフォルトのページサイズは 500 なので、異なるサイズを試して何が最適かを見つけてください。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内のすべての run のメトリクスを CSV ファイルにエクスポートする

このスクリプトは、プロジェクト内の run を取得し、その名前、設定、サマリー統計を含むデータフレームと CSV を生成します。`<entity>` と `<project>` をあなたの W&B entity とプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary はメトリクス（例: 精度）の出力キー/値を含む
    #  ._json_dict を呼び出して大きなファイルを省略します
    summary_list.append(run.summary._json_dict)

    # .config はハイパーパラメーターを含む
    #  特殊な値（_で始まる）を削除
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は人間が読める run の名前
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run の開始時間を取得する

このコードスニペットは、run が作成された時刻を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 終了した run にファイルをアップロードする

以下のコードスニペットは、選択されたファイルを終了した run にアップロードします。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### run からファイルをダウンロードする

これは、cifar プロジェクトの run ID uxte44z7 に関連付けられた "model-best.h5" ファイルを見つけ、それをローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### run からすべてのファイルをダウンロードする

これは、run に関連するすべてのファイルを見つけ、それをローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定の Sweep から run を取得する

このスニペットは、特定の Sweep に関連するすべての run をダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweep から最良の run を取得する

以下のスニペットは、指定された Sweep から最良の run を取得します。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は、sweep 設定で `metric` パラメータによって定義された最良のメトリクスを持つ run です。

### Sweep から最高のモデルファイルをダウンロードする

このスニペットは、`model.h5` にモデルファイルを保存した run がある Sweep から最高の検証精度のモデルファイルをダウンロードします。

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

### run から指定された拡張子のファイルをすべて削除する

このスニペットは、run から指定された拡張子のファイルを削除します。

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

### システムメトリクスデータをダウンロードする

このスニペットは、run のすべてのシステムリソース消費メトリクスを含むデータフレームを作成し、それを CSV に保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サマリーメトリクスを更新する

辞書を渡してサマリーメトリクスを更新できます。

```python
summary.update({"key": val})
```

### run を実行したコマンドを取得する

各 run は、run 概要ページでそれを実行したコマンドをキャプチャします。このコマンドを API から取得するには、次のように実行できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```