---
title: データのインポートとエクスポート
description: MLFlow からデータをインポートしたり、W&B に保存したデータをエクスポート・更新したりできます。
menu:
  default:
    identifier: public-api-guide
    parent: experiments
weight: 8
---

W&B の Public API を使ってデータのエクスポートやインポートを行う方法について説明します。

{{% alert %}}
この機能は python>=3.8 が必要です
{{% /alert %}}

## MLFlow からデータをインポートする

W&B は MLFlow からの実験、runs、artifacts、メトリクス、その他メタデータのインポートに対応しています。

依存関係をインストールします:

```shell
# 注意: py38+ が必要です
pip install wandb[importers]
```

W&B にログインします。初めて利用する場合は、表示される指示に従ってください。

```shell
wandb login
```

既存の MLFlow サーバーから全ての runs をインポートします:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` は MLFlow サーバーの全ての runs を収集します。特定のサブセットだけアップロードしたい場合は、自分で runs のイテラブルを作成してインポーターに渡すことも可能です。

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
Databricks MLFlow からインポートする場合は、事前に [Databricks CLI の設定](https://docs.databricks.com/dev-tools/cli/index.html) が必要な場合があります。

この場合は、前のステップで `mlflow-tracking-uri="databricks"` を設定してください。
{{% /alert %}}

artifacts のインポートをスキップしたい場合、`artifacts=False` を指定できます:

```py
importer.import_runs(runs, artifacts=False)
```

特定の W&B Entity や Project にインポートしたい場合は、`Namespace` を使用します:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## データのエクスポート

W&B に保存したデータを Public API を使ってエクスポートやアップデートできます。この API を使う前に、スクリプトからデータをログしておく必要があります。詳細は [クイックスタート]({{< relref "/guides/quickstart.md" >}}) を参照してください。

**Public API のユースケース**

- **データのエクスポート**: Jupyter Notebook でカスタム分析のためにデータフレームを取得できます。データを探索した後、分析用の run を作成して結果をログすることで学びを同期できます（例: `wandb.init(job_type="analysis")`）。
- **既存の Run のアップデート**: W&B の run に紐付けてログしたデータを更新できます。例えば、複数の run の config に新しい情報（アーキテクチャーや当初ログしていなかったハイパーパラメーターなど）を追加したい場合に有用です。

利用可能な関数の詳細は [API リファレンス]({{< relref "/ref/python/public-api/" >}}) を参照してください。

### API キーを作成する

APIキーはあなたのマシンをW&Bに認証します。ユーザープロファイルから生成できます。

{{% alert %}}
より手軽な方法として、[W&B認証ページ](https://wandb.ai/authorize)に直接アクセスしてAPIキーを生成できます。表示されたAPIキーはパスワードマネージャーなど安全な場所に保管してください。
{{% /alert %}}

1. 画面右上のユーザープロファイルアイコンをクリックします。
2. **User Settings** を選び、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックし、表示されたAPIキーをコピーしてください。APIキーを非表示にしたい場合はページを再読み込みしてください。

### run のパスを調べる

Public API を使う際には `<entity>/<project>/<run_id>` という run のパスがよく必要になります。アプリの UI 上で run ページを開き、[Overviewタブ]({{< relref "/guides/models/track/runs/#overview-tab" >}}) をクリックすると run パスが表示されます。

### Run データのエクスポート

完了済みやアクティブな run からデータをダウンロードできます。よくある用途としては、カスタム分析のためにJupyterノートブックへデータフレームをダウンロードしたり、自動化環境で独自処理に使うパターンがあります。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run オブジェクトでよく使われる属性は以下の通りです。

| 属性              | 意味                                                                                                                                                                                                                                                                                                                                             |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `run.config`      | run の設定情報（例: トレーニングrunならハイパーパラメーター、データセット Artifact 作成runなら前処理メソッドなど）の辞書。run の入力にあたります。                                                                                                                                                                                                |
| `run.history()`   | トレーニング中に値が変化するもの（例: loss など）を辞書のリストとして格納。`run.log()` コマンドでこの中に値が追加されます。                                                                                                                                                                                                                      |
| `run.summary`     | run の結果をまとめた情報の辞書。精度やlossなどのスカラーや大きいファイルが含まれます。`run.log()` はデフォルトで summary をログ済み時系列の最終値に設定しますが、直接設定も可能。summary は run の出力と考えてください。                                                                                                           |

過去の run のデータも編集・更新できます。デフォルトでは、APIオブジェクトの単一インスタンスは全てのネットワークリクエストをキャッシュします。リアルタイムで更新が必要な場合は `api.flush()` を呼び出してください。

### 各 run 属性の詳細

以下は run を作成し、データをログし、各属性へアクセスする例です。

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

このrunの各属性の出力例を以下に示します。

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

デフォルトの history メソッドでは、メトリクスは一定数（デフォルト500件。`samples` 引数で変更可）にサンプリングされます。大規模な run の全データをエクスポートしたい場合は `run.scan_history()` を使ってください。[APIリファレンス]({{< relref "/ref/python/public-api" >}}) も参照ください。

### 複数の Run を取得・検索

{{< tabpane text=true >}}
    {{% tab header="DataFrameとCSV" %}}
このスクリプトは project を特定し name, config, summary stats を含む run の CSV を出力します。`<entity>`と`<project>`をあなたのW&B entityとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary にはaccuracyなど出力系のメトリクスが入っています
    #  ._json_dictを使い大きなファイルは除外します
    summary_list.append(run.summary._json_dict)

    # .config はハイパーパラメーター
    #  _ で始まる特殊値は除外します
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameはrunの可読名
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")

run.finish()
```
    {{% /tab %}}
    {{% tab header="MongoDB形式" %}}
W&B API は `api.runs()` でプロジェクト内の runs を横断的に検索する方法も提供しています。最も一般的な用途は、カスタム分析用に runs データをエクスポートすることです。クエリインターフェースは [MongoDB公式のクエリ構文](https://docs.mongodb.com/manual/reference/operator/query)と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}

`api.runs` はイテラブルな `Runs` オブジェクトを返し、リストのように扱えます。デフォルトでは50件ずつ順次ロードしますが、`per_page` キーワード引数で1ページあたりの数を変更できます。

`api.runs` は `order` 引数も使えます。デフォルトは `-created_at` です。昇順にしたい場合は `+created_at` を指定してください。configやsummaryの値でソートも可能です（例: `summary.val_acc` や `config.experiment_name`）。

### エラーハンドリング

W&Bサーバーとの通信中にエラーが発生した場合、`wandb.CommError` が発生します。元の例外は `exc` 属性で参照可能です。

### API で最新の git commit を取得する

UI 上では run をクリック後、run ページの Overviewタブ で最新の git commit を確認できます。また、`wandb-metadata.json` ファイルにも記録されています。Public API を使う場合は `run.commit` で hash を取得できます。

### 実行中に run の name や ID を取得する

`wandb.init()` 呼び出し後、スクリプト内から run のランダムIDや可読名にアクセスできます。

- 一意な run ID（8文字のハッシュ）: `run.id`
- ランダムで可読な run name: `run.name`

識別子を活用するコツを挙げます:

- **Run ID**: 自動生成されたハッシュのままにしましょう。Project 内で一意である必要があります。
- **Run name**: 短く・読みやすく・できれば一意な文字列にして、グラフ上でもrunを区別しやすくしましょう。
- **Run notes**: そのrunで何をしたかの簡単なメモ置き場として最適です。`wandb.init(notes="ここにメモ")` で設定可能です。
- **Run tags**: run tag で特徴を動的に管理・UI の filter で目的の run のみに絞り込めます。タグはスクリプトからもUI（runテーブルや Overviewタブ）からも編集できます。詳しい操作は[こちら]({{< relref "/guides/models/track/runs/tags.md" >}})。

## Public API のサンプル

### matplotlib や seaborn で可視化するためにデータをエクスポート

よく使われるエクスポートパターンは[APIサンプル]({{< relref "/ref/python/public-api/" >}})を参照してください。カスタムプロットやrunテーブル上でダウンロードボタンをクリックして CSV 形式でデータを取得することもできます。

### Run からメトリクスを読み込む

この例では `run.log({"accuracy": acc})` で保存した timestamp と accuracy を出力します。対象 run は `"<entity>/<project>/<run_id>"` です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Runs をフィルタリングする

MongoDBクエリ言語で runs を絞り込めます。

#### 日付で絞る

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### Run から特定メトリクスのみを取得

特定のメトリクスのみ取得したい場合、`keys` 引数を使います。`run.history()`利用時はデフォルトで500件サンプリングされます。対象キーが含まれない step は dataframe 内で `NaN` になります。`keys` 引数は指定したメトリクスキーでサンプルが優先されます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2つの Run を比較する

この例では `run1` と `run2` 間で異なる config パラメータを出力します。

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>, <project>, <run_id> を置き換えてください
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

### Run 完了後にメトリクスを更新する

この例では既存 run の精度 (`accuracy`) を `0.9` に設定し、既存 run の accuracy ヒストグラムを numpy_array のヒストグラムに置き換えます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 完了済み Run でメトリクス名を変更する

この例では summary カラムの名前を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
カラム名の変更は テーブル のみに適用されます。Charts では元のメトリクス名で参照されます。
{{% /alert %}}



### 既存 Run の config を更新する

この例は既存の設定項目を1つ更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース使用量を CSV にエクスポート

このスニペットでシステムリソース使用状況を見つけてCSVに保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 未サンプリングのメトリクスデータを取得

history からデータを取り出すとデフォルトでは500点サンプリングになります。`run.scan_history()` を使うとログされた全データポイントを取得できます。以下は `loss` ログの全データをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history からページ分割でデータを取得

バックエンドでメトリクス取得が遅い場合やAPIリクエストがタイムアウトする場合は、`scan_history` のページサイズを下げてみてください。デフォルトは500件です。状況に応じて数値を調整できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内の全 run からメトリクスを CSV でエクスポート

プロジェクト内全 run を取得し、name, config, summary stats を含むデータフレームと CSV を出力します。`<entity>`と`<project>`をあなたのW&B entityとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には accuracy など出力値
    #  ._json_dict で大きなファイルを除外
    summary_list.append(run.summary._json_dict)

    # .config はハイパーパラメーター
    #  _ で始まる特殊値は除外
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name はrunの可読名
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run の開始時刻を取得する

このコードは run が作成された時間を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了済み Run にファイルをアップロード

以下の例で任意のファイルを完了済みRunにアップロードできます。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### Run からファイルをダウンロード

この例では `cifar` プロジェクトの run ID uxte44z7 に紐づく "model-best.h5" ファイルをローカル保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### Run から全ファイルをダウンロード

該当ランに紐づく全てのファイルをローカル保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 指定 Sweep の全 Runs を取得

特定の sweep に紐づく全 run をダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweep でベストな Run を取得

指定した sweep における最高の run を取得するスニペットです。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は sweep config の metric パラメータで定義された最良 run です。

### Sweep の中でベストなモデルファイルをダウンロード

この例では Sweep 内で validation accuracy が最も高い run の `model.h5` をダウンロードします。

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

### Run から特定拡張子のファイルを削除

指定拡張子を持つファイルを run から削除する例です。

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

### システムメトリクスデータのダウンロード

run の全システムリソース使用量メトリクスを DataFrame化し、CSVとして保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### summary メトリクスの更新

summaryメトリクスは辞書形式で更新できます。

```python
summary.update({"key": val})
```

### run を実行したコマンドを取得

各 run では Overview ページに使用したコマンドが表示されていますが、API からも直接取得できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```