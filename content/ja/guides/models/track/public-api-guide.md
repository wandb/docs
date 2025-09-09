---
title: データのインポートとエクスポート
description: MLFlow からデータをインポートし、W&B に保存したデータをエクスポートまたは更新します
menu:
  default:
    identifier: ja-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

W&B Public API を使ってデータをエクスポートまたはインポートします。
{{% alert %}}
この機能を利用するには Python>=3.8 が必要です
{{% /alert %}}

## MLFlow からデータをインポートする

W&B は MLFlow からのデータのインポートをサポートしており、experiments、runs、artifacts、メトリクス、その他のメタデータを含みます。

依存関係をインストールします:

```shell
# 注: これは py38+ が必要です
pip install wandb[importers]
```

W&B にログインします。初めてログインする場合はプロンプトに従ってください。

```shell
wandb login
```

既存の MLFlow サーバーからすべての runs をインポートします:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` は MLFlow サーバーからすべての runs を収集します。特定のサブセットだけをアップロードしたい場合は、独自の runs のイテラブルを作成してインポーターに渡せます。

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
Databricks MLFlow からインポートする場合は、先に [Databricks CLI を設定](https://docs.databricks.com/dev-tools/cli/index.html) する必要があるかもしれません。

前のステップで `mlflow-tracking-uri="databricks"` を設定してください。
{{% /alert %}}

artifacts のインポートをスキップするには、`artifacts=False` を渡します:

```py
importer.import_runs(runs, artifacts=False)
```

特定の W&B entity と project にインポートするには、`Namespace` を渡します:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```




## データのエクスポート

Public API を使って、W&B に保存したデータのエクスポートや更新ができます。この API を使う前に、スクリプトからデータをログしておいてください。詳しくは [クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) を参照してください。

**Public API のユースケース**

- **データのエクスポート**: Jupyter Notebook でのカスタム分析のために DataFrame を取得します。データを探索したら、新しい分析用の run を作成して結果をログし、学びを同期できます。例: `wandb.init(job_type="analysis")`
- **既存の Runs を更新**: W&B の run に紐づいてログ済みのデータを更新できます。たとえば、当初はログしていなかったアーキテクチャーやハイパーパラメーターなどの追加情報を含めるために、一連の runs の config を更新したい場合などです。

利用可能な関数の詳細は [Generated Reference Docs]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。

### APIキー を作成する

APIキー は、あなたのマシンを W&B に認証します。APIキー はユーザープロファイルから作成できます。

{{% alert %}}
より手早い方法としては、[W&B authorization page](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を隠すにはページを再読み込みします。


### run パスを見つける

Public API を使う際は、`<entity>/<project>/<run_id>` の形式の run パスが必要になることがよくあります。アプリの UI で run ページを開き、[Overviewタブ ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})をクリックして run パスを取得します。


### Run データをエクスポートする

完了済みまたは実行中の run からデータをダウンロードします。一般的な用途には、Jupyter Notebook でのカスタム分析用に DataFrame をダウンロードすることや、自動化された環境で独自のロジックを使うことなどがあります。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run オブジェクトで最もよく使われる属性は次のとおりです:

| Attribute       | Meaning                                                                                                                                                                                                                                                                                                              |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | run の設定情報を表す辞書。トレーニング run のハイパーパラメーターや、データセット Artifact を作成する run の前処理方法などが含まれます。run の入力だと考えてください。                                                                                                    |
| `run.history()` | モデルのトレーニング中に変化する値（例: loss）を保存するための辞書のリスト。`run.log()` コマンドはこのオブジェクトに追記します。                                                                                                                                                                 |
| `run.summary`   | run の結果を要約する情報の辞書。accuracy や loss のようなスカラー値や大きなファイルを含めることができます。デフォルトでは、`run.log()` はログされた時系列の最終値を summary に設定します。summary の内容は直接設定することもできます。summary は run の出力だと考えてください。 |

過去の runs のデータを変更または更新することもできます。デフォルトでは、単一の API オブジェクトインスタンスはすべてのネットワークリクエストをキャッシュします。実行中のスクリプトでリアルタイムの情報が必要なユースケースでは、`api.flush()` を呼び出して最新の値を取得してください。

### さまざまな run 属性を理解する

以下のコードスニペットは、run を作成し、いくつかのデータをログしてから、その run の属性にアクセスする方法を示します:

```python
import wandb
import random

with wandb.init(project="public-api-example") as run:
    n_epochs = 5
    config = {"n_epochs": n_epochs}
    run.config.update(config)
    for n in range(run.config.get("n_epochs")):
        run.log(
            {"val": random.randint(0, 1000), "loss": (random.randint(0, 1000) / 1000.00)}
        )
```

以下のセクションでは、上記の run オブジェクト属性の出力の違いを説明します

##### `run.config`

```python
{"n_epochs": 5}
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

デフォルトの history メソッドは、メトリクスを固定数のサンプル（デフォルトは 500、`samples` 引数で変更可能）に間引きます。大きな run のデータをすべてエクスポートしたい場合は、`run.scan_history()` メソッドを使ってください。詳しくは [API リファレンス]({{< relref path="/ref/python/public-api" lang="ja" >}}) を参照してください。

### 複数の Runs をクエリする

{{< tabpane text=true >}}
    {{% tab header="DataFrame と CSVs" %}}
このサンプルスクリプトは Project を見つけ、name、config、summary の統計を含む runs の CSV を出力します。`<entity>` と `<project>` を、それぞれあなたの W&B entity と Project 名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には accuracy などのメトリクスに関する
    # 出力のキーと値が含まれます。
    # 大きなファイルを除外するために ._json_dict を呼び出します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれます。
    # 先頭が _ の特別な値は除外します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は人間が読みやすい run の名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")

run.finish()
```
    {{% /tab %}}
    {{% tab header="MongoDB 形式" %}}
W&B API は、`api.runs()` を使って Project 内の runs を横断的にクエリする方法も提供します。最も一般的なユースケースは、カスタム分析のために runs のデータをエクスポートすることです。クエリインターフェースは [MongoDB が採用しているもの](https://docs.mongodb.com/manual/reference/operator/query) と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}


`api.runs` を呼び出すと、反復可能でリストのように振る舞う `Runs` オブジェクトが返ります。デフォルトでは、このオブジェクトは必要に応じて 1 ページにつき 50 件の run を順番にロードしますが、`per_page` キーワード引数で 1 ページの件数を変更できます。

`api.runs` は `order` キーワード引数も受け取ります。デフォルトの順序は `-created_at` です。昇順に並べるには `+created_at` を指定します。config や summary の値でソートすることもできます。例えば、`summary.val_acc` や `config.experiment_name` です。

### エラーハンドリング

W&B サーバーとの通信中にエラーが発生した場合、`wandb.CommError` が送出されます。元の例外は `exc` 属性から調査できます。

### API 経由で最新の git commit を取得する

UI で run をクリックし、run ページの Overview タブをクリックすると、最新の git commit が表示されます。これは `wandb-metadata.json` にも含まれています。Public API を使うと、`run.commit` で git ハッシュを取得できます。

### 実行中に run の名前と ID を取得する

`wandb.init()` を呼び出した後、スクリプトからランダムな run ID や人間が読みやすい run 名にアクセスできます:

- 一意の run ID（8 文字のハッシュ）: `run.id`
- ランダムな run 名（人間が読みやすい）: `run.name`

runs に便利な識別子を設定する方法を検討している場合は、以下をおすすめします:

- **Run ID**: 生成されたハッシュのままにします。これは Project 内の runs で一意である必要があります。
- **Run name**: 短く読みやすく、可能であれば一意にして、チャート上の異なる線を見分けられるようにします。
- **Run notes**: run で何をしているかの簡単な説明を入れるのに最適な場所です。`wandb.init(notes="your notes here")` で設定できます。
- **Run tags**: run のタグで動的にトラッキングし、UI のフィルターを使って関心のある runs のみにテーブルを絞り込みます。タグはスクリプトから設定でき、UI の runs テーブルや run ページの Overview タブでも編集できます。詳しい手順は[こちら]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})を参照してください。

## Public API のサンプル

### matplotlib や seaborn で可視化するためにデータをエクスポートする

一般的なエクスポートパターンについては [API のサンプル]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。カスタムプロットや展開した runs テーブル上のダウンロードボタンをクリックして、ブラウザから CSV をダウンロードすることもできます。

### run からメトリクスを読み取る

この例は、`"<entity>/<project>/<run_id>"` に保存された run について、`run.log({"accuracy": acc})` で保存された timestamp と accuracy を出力します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### runs をフィルターする

MongoDB Query Language を使ってフィルターできます。

#### 日付

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### run から特定のメトリクスを読む

run から特定のメトリクスを取得するには、`keys` 引数を使用します。`run.history()` のデフォルトのサンプル数は 500 です。特定のメトリクスを含まないステップは、出力のデータフレームでは `NaN` として表示されます。`keys` 引数を指定すると、指定したメトリクスキーを含むステップが優先的にサンプリングされます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2 つの runs を比較する

これは、`run1` と `run2` の間で異なる config パラメータを出力します。

```python
import pandas as pd
import wandb

api = wandb.Api()

# あなたの <entity>, <project>, <run_id> に置き換えてください
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

### run 完了後にメトリクスを更新する

この例では、過去の run の accuracy を `0.9` に設定します。また、過去の run の accuracy のヒストグラムを `numpy_array` のヒストグラムに変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 完了した run のメトリクス名を変更する

この例は、Tables の summary 列の名前を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
列名の変更は Tables にのみ適用されます。チャートは元の名前でメトリクスを参照し続けます。
{{% /alert %}}



### 既存の run の config を更新する

この例は設定の一部を更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース使用量を CSV ファイルにエクスポートする

以下のスニペットはシステムリソースの使用量を取得し、CSV に保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サンプリングされていないメトリクスデータを取得する

history からデータを取得すると、デフォルトでは 500 ポイントにサンプリングされます。`run.scan_history()` を使って、ログされたすべてのデータポイントを取得できます。以下は、history にログされたすべての `loss` データポイントをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history からページネーションされたデータを取得する

バックエンドでのメトリクスの取得が遅い、または API リクエストがタイムアウトする場合は、`scan_history` のページサイズを下げて、個々のリクエストがタイムアウトしないようにしてみてください。デフォルトのページサイズは 500 なので、最適なサイズをいくつか試してみてください:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### Project 内のすべての runs からメトリクスを CSV にエクスポートする

このスクリプトは Project 内の runs を取得し、name、config、summary の統計を含むデータフレームと CSV を生成します。`<entity>` と `<project>` を、それぞれあなたの W&B entity と Project 名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には出力のキー/値、
    #  たとえば accuracy などのメトリクスが含まれます。
    #  大きなファイルを除外するために ._json_dict を呼び出します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれます。
    #  先頭が _ の特別な値は除外します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は人間が読みやすい run の名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run の開始時刻を取得する

このコードスニペットは、run が作成された時刻を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了した run にファイルをアップロードする

以下のコードスニペットは、選択したファイルを完了した run にアップロードします。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### run からファイルをダウンロードする

これは cifar Project の run ID uxte44z7 に関連付けられた "model-best.h5" ファイルを見つけて、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### run からすべてのファイルをダウンロードする

これは run に関連するすべてのファイルを見つけ、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定の Sweep から runs を取得する

このスニペットは、特定の Sweep に関連付けられたすべての runs をダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweep からベストな run を取得する

次のスニペットは、指定した Sweep から最良の run を取得します。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は、Sweep の config の `metric` パラメータで定義された最良のメトリクスを持つ run です。

### Sweep からベストな model ファイルをダウンロードする

このスニペットは、runs が `model.h5` に model ファイルを保存している Sweep から、検証精度が最も高い model ファイルをダウンロードします。

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

### 指定した拡張子のファイルを run からすべて削除する

このスニペットは、指定した拡張子を持つファイルを run から削除します。

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

このスニペットは、run のすべてのシステムリソース使用メトリクスを含むデータフレームを生成し、CSV に保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### summary メトリクスを更新する

辞書を渡して summary メトリクスを更新できます。

```python
summary.update({"key": val})
```

### run を実行したコマンドを取得する

各 run では、run の Overview ページに起動したコマンドが保存されています。API からこのコマンドを取得するには、次を実行します:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```