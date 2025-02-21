---
title: Import and export data
description: MLFlow からデータをインポートしたり、W&B に保存したデータをエクスポートまたは更新したりできます。
menu:
  default:
    identifier: ja-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

W&B Public API を使用して、データのエクスポートまたはインポートを行います。

{{% alert %}}
この機能には python>=3.8 が必要です。
{{% /alert %}}

## MLFlow からのデータインポート

W&B は、実験、run、Artifacts、メトリクス、その他のメタデータなど、MLFlow からのデータインポートをサポートしています。

依存関係をインストールします。

```shell
# note: this requires py38+
pip install wandb[importers]
```

W&B にログインします。まだログインしていない場合は、プロンプトに従ってください。

```shell
wandb login
```

既存の MLFlow サーバーからすべての run をインポートします。

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` は MLFlow サーバーからすべての run を収集します。特定のサブセットをアップロードする場合は、独自の runs イテラブルを構築してインポーターに渡すことができます。

```py
import mlflow
from wandb.apis.importers.mlflow import MlflowRun

client = mlflow.tracking.MlflowClient(mlflow_tracking_uri)

runs: Iterable[MlflowRun] = []
for run in mlflow_client.search_runs(...):
    runs.append(MlflowRun(run, client))

importer.import_runs(runs)
```

{{% alert %}}
Databricks MLFlow からインポートする場合は、最初に [Databricks CLI を構成する](https://docs.databricks.com/dev-tools/cli/index.html) 必要がある場合があります。

前のステップで `mlflow-tracking-uri="databricks"` を設定します。
{{% /alert %}}

Artifacts のインポートをスキップするには、`artifacts=False` を渡します。

```py
importer.import_runs(runs, artifacts=False)
```

特定の W&B の entity および project にインポートするには、`Namespace` を渡します。

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## データのエクスポート

Public API を使用して、W&B に保存したデータのエクスポートまたは更新を行います。この API を使用する前に、スクリプトからデータをログに記録してください。詳細については、[クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}})を確認してください。

**Public API のユースケース**

- **データのエクスポート**: Jupyter Notebook でのカスタム分析用にデータフレームをプルダウンします。データを調べたら、新しい分析 run を作成して結果をログに記録することで、調査結果を同期できます。例: `wandb.init(job_type="analysis")`
- **既存の Runs の更新**: W&B の run に関連してログに記録されたデータを更新できます。たとえば、アーキテクチャや、最初にログに記録されなかったハイパーパラメーターなど、追加情報を含めるように一連の runs の設定を更新する場合があります。

利用可能な機能の詳細については、[生成されたリファレンスドキュメント]({{< relref path="/ref/python/public-api/" lang="ja" >}})を参照してください。

### API キーの作成

API キー は、W&B に対するマシンの認証を行います。API キーは、ユーザープロファイルから生成できます。

{{% alert %}}
より効率的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キー を生成できます。表示された API キーをコピーして、パスワードマネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にあるユーザープロファイルアイコンをクリックします。
2. **ユーザー設定** を選択し、**API キー** セクションまでスクロールします。
3. **表示** をクリックします。表示された API キー をコピーします。API キー を非表示にするには、ページをリロードします。

### run パスを見つける

Public API を使用するには、`<entity>/<project>/<run_id>` である run パスが必要になることがよくあります。アプリの UI で、run ページを開き、[Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) をクリックして run パスを取得します。

### Run データのエクスポート

完了した run またはアクティブな run からデータをダウンロードします。一般的な使用法としては、Jupyter Notebook でのカスタム分析用にデータフレームをダウンロードしたり、自動化された環境でカスタムロジックを使用したりすることがあります。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run オブジェクトの最も一般的に使用される属性は次のとおりです。

| 属性          | 意味                                                                                                                                                                                                                                                                                       |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run.config`    | トレーニング run のハイパーパラメーターや、データセット Artifacts を作成する run のプリプロセッシングメソッドなど、run の構成情報の辞書。これらは run の入力と考えることができます。                                                                                                                                                           |
| `run.history()` | 損失など、モデルのトレーニング中に変化する値を保存するための辞書のリスト。コマンド `wandb.log()` はこのオブジェクトに追加します。                                                                                                                                                            |
| `run.summary`   | run の結果を要約する情報の辞書。これは、精度や損失などのスカラー、または大きなファイルにすることができます。デフォルトでは、`wandb.log()` はサマリーをログに記録された時系列の最終値に設定します。サマリーの内容は直接設定することもできます。サマリーは run の出力と考えることができます。 |

過去の runs のデータを変更または更新することもできます。デフォルトでは、api オブジェクトの単一インスタンスはすべてのネットワークリクエストをキャッシュします。ユースケースで実行中のスクリプトでリアルタイムの情報が必要な場合は、`api.flush()` を呼び出して更新された値を取得します。

### さまざまな属性について

以下の run について

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

これらは上記の run オブジェクト属性のさまざまな出力です。

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

デフォルトの history メソッドは、メトリクスを固定数のサンプルにサンプリングします（デフォルトは 500 です。これは `samples` __ 引数で変更できます）。大規模な run でデータをすべてエクスポートする場合は、`run.scan_history()` メソッドを使用できます。詳細については、[API リファレンス]({{< relref path="/ref/python/public-api" lang="ja" >}})を参照してください。

### 複数の Runs のクエリ

{{< tabpane text=true >}}
    {{% tab header="DataFrame と CSV" %}}
このサンプルスクリプトは、project を検索し、名前、設定、サマリー統計を含む runs の CSV を出力します。`<entity>` と `<project>` を、それぞれ W&B の entity と project の名前に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には、精度などの
    # メトリクスの出力キーと値が含まれています。
    #  大きなファイルを省略するために、._json_dict を呼び出します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれています。
    #  _ で始まる特殊な値を削除します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は、run の人間が読める名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```
    {{% /tab %}}
    {{% tab header="MongoDB スタイル" %}}
W&B API は、api.runs() を使用して project 内の runs に対してクエリを実行する方法も提供します。最も一般的なユースケースは、カスタム分析用に runs データをエクスポートすることです。クエリインターフェースは、[MongoDB が使用するインターフェース](https://docs.mongodb.com/manual/reference/operator/query) と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```
    {{% /tab %}}
{{< /tabpane >}}

`api.runs` を呼び出すと、イテラブルでリストのように動作する `Runs` オブジェクトが返されます。デフォルトでは、オブジェクトは必要に応じて一度に 50 個の runs を順番にロードしますが、`per_page` キーワード引数を使用して、ページごとにロードされる数を変更できます。

`api.runs` は、`order` キーワード引数も受け入れます。デフォルトの順序は `-created_at` です。結果を昇順で並べるには、`+created_at` を指定します。設定またはサマリー値で並べ替えることもできます。たとえば、`summary.val_acc` または `config.experiment_name` などです。

### エラー処理

W&B サーバーとの通信中にエラーが発生すると、`wandb.CommError` が発生します。元の例外は、`exc` 属性を介して調べることができます。

### API 経由で最新の git コミットを取得する

UI で、run をクリックし、run ページの [Overview タブ] をクリックして、最新の git コミットを表示します。これは、ファイル `wandb-metadata.json` にもあります。Public API を使用すると、`run.commit` で git ハッシュを取得できます。

### run の実行中に run の名前と ID を取得する

`wandb.init()` を呼び出した後、スクリプトからランダムな run ID または人間が読める run 名に次のようにアクセスできます。

- 一意の run ID (8 文字のハッシュ): `wandb.run.id`
- ランダムな run 名 (人間が読める): `wandb.run.name`

runs に役立つ識別子を設定する方法を検討している場合は、以下をお勧めします。

- **Run ID**: 生成されたハッシュのままにします。これは、project 内の runs 間で一意である必要があります。
- **Run 名**: これは、チャート上の異なる行を区別できるように、短く、読みやすく、できれば一意である必要があります。
- **Run ノート**: これは、run で行っていることの簡単な説明を記述するのに最適な場所です。これは、`wandb.init(notes="ここにノート")` で設定できます
- **Run タグ**: run タグで動的に追跡し、UI のフィルターを使用して、関心のある runs だけにテーブルを絞り込みます。スクリプトからタグを設定し、runs テーブルと run ページの Overview タブの両方で UI で編集できます。詳細な手順については、[こちら]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})を参照してください。

## Public API の例

### データをエクスポートして matplotlib または seaborn で視覚化する

一般的なエクスポートパターンの例については、[API の例]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。カスタムプロットまたは拡張された runs テーブルのダウンロードボタンをクリックして、ブラウザーから CSV をダウンロードすることもできます。

### Run からメトリクスを読み取る

この例では、`"<entity>/<project>/<run_id>"` に保存された run に対して `wandb.log({"accuracy": acc})` で保存されたタイムスタンプと精度を出力します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Runs のフィルタリング

MongoDB Query Language を使用してフィルタリングできます。

#### 日付

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### Run から特定のメトリクスを読み取る

run から特定のメトリクスをプルするには、`keys` 引数を使用します。`run.history()` を使用する場合のデフォルトのサンプル数は 500 です。特定のメトリクスを含まないログに記録されたステップは、出力データフレームに `NaN` として表示されます。`keys` 引数を指定すると、API はリストされたメトリクスキーを含むステップをより頻繁にサンプリングします。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2 つの Runs を比較する

これにより、`run1` と `run2` で異なる設定パラメーターが出力されます。

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>、<project>、<run_id> に置き換えます
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")


df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

出力:

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### run の完了後、run のメトリクスを更新する

この例では、以前の run の精度を `0.9` に設定します。また、以前の run の精度ヒストグラムを `numpy_array` のヒストグラムに変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 完了した run でメトリクスの名前を変更する

この例では、テーブルのサマリー列の名前を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
列の名前の変更は、テーブルにのみ適用されます。チャートでは、メトリクスは元の名前で参照されます。
{{% /alert %}}

### 既存の run の設定を更新する

この例では、構成設定の 1 つを更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソースの消費量を CSV ファイルにエクスポートする

以下のスニペットは、システムリソースの消費量を見つけ、CSV に保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サンプリングされていないメトリクスデータを取得する

history からデータをプルすると、デフォルトで 500 ポイントにサンプリングされます。`run.scan_history()` を使用して、ログに記録されたすべてのデータポイントを取得します。history にログに記録されたすべての `loss` データポイントをダウンロードする例を次に示します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history からページ分割されたデータを取得する

バックエンドでメトリクスの取得が遅い場合、または API リクエストがタイムアウトする場合は、個々のリクエストがタイムアウトしないように、`scan_history` のページサイズを小さくしてみてください。デフォルトのページサイズは 500 なので、さまざまなサイズを試して最適なサイズを確認できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### project 内のすべての Runs からメトリクスを CSV ファイルにエクスポートする

このスクリプトは、project 内の runs をプルダウンし、名前、設定、サマリー統計を含む runs のデータフレームと CSV を生成します。`<entity>` と `<project>` を、それぞれ W&B の entity と project の名前に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には、精度などの
    #  メトリクスの出力キーと値が含まれています。
    #  大きなファイルを省略するために、._json_dict を呼び出します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれています。
    #  _ で始まる特殊な値を削除します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は、run の人間が読める名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### Run の開始時刻を取得する

このコードスニペットは、run が作成された時刻を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了した Run にファイルをアップロードする

以下のコードスニペットは、選択したファイルを完了した run にアップロードします。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### Run からファイルをダウンロードする

これは、cifar project の run ID uxte44z7 に関連付けられたファイル "model-best.h5" を検索し、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### Run からすべてのファイルをダウンロードする

これは、run に関連付けられたすべてのファイルを検索し、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定の sweep から Runs を取得する

このスニペットは、特定の sweep に関連付けられたすべての runs をダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweep から最適な Run を取得する

次のスニペットは、指定された sweep から最適な run を取得します。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は、sweep 設定の `metric` パラメーターで定義されているように、最適なメトリクスを持つ run です。

### Sweep から最適なモデルファイルをダウンロードする

このスニペットは、モデルファイルを `model.h5` に保存した runs を持つ sweep から、検証精度が最も高いモデルファイルをダウンロードします。

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

### Run から特定の拡張子を持つすべてのファイルを削除する

このスニペットは、run から特定の拡張子を持つファイルを削除します。

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

このスニペットは、run のすべてのシステムリソース消費量メトリクスを含むデータフレームを生成し、CSV に保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サマリーメトリクスの更新

辞書を渡して、サマリーメトリクスを更新できます。

```python
summary.update({"key": val})
```

### Run を実行したコマンドを取得する

各 run は、run の Overview ページで起動したコマンドをキャプチャします。API からこのコマンドをプルダウンするには、次を実行します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```
