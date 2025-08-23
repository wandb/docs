---
title: データのインポートとエクスポート
description: MLFlow からデータをインポートしたり、W&B に保存したデータをエクスポートまたは更新したりできます
menu:
  default:
    identifier: ja-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

W&B のパブリック API を使ってデータをエクスポート、またはインポートします。

{{% alert %}}
この機能は python>=3.8 が必要です。
{{% /alert %}}

## MLFlow からデータをインポート

W&B は MLFlow からのデータインポートに対応しています。実験、Runs、Artifacts、メトリクス、その他メタデータも含まれます。

依存関係をインストールします：

```shell
# 注意: これは py38+ が必要です
pip install wandb[importers]
```

W&B にログインします。初回の場合はプロンプトに従ってログインしてください。

```shell
wandb login
```

既存の MLFlow サーバーからすべての Run をインポート：

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトで `importer.collect_runs()` は MLFlow サーバー上のすべての Run を収集します。特定のサブセットだけをアップロードしたい場合は、自分で runs のイテラブルを作成し、importer に渡せます。

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
Databricks MLFlow からインポートする場合は、事前に [Databricks CLI の設定が必要です](https://docs.databricks.com/dev-tools/cli/index.html)。

前述の `mlflow-tracking-uri="databricks"` を設定してください。
{{% /alert %}}

Artifacts のインポートをスキップしたい場合は、`artifacts=False` を渡せます：

```py
importer.import_runs(runs, artifacts=False)
```

特定の W&B entity と project にインポートする場合は、`Namespace` を指定できます：

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```



## データのエクスポート

Public API を使って、W&B に保存されたデータをエクスポートまたはアップデートできます。API を利用する前に、まずスクリプトからデータをログしてください。[クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}}) で詳細を確認できます。

**Public API の主なユースケース**

- **データのエクスポート**: Jupyter Notebook で独自に分析するためにデータフレームを取得できます。データ探索後は、新しい analysis Run を作成し、結果をログとして同期も可能です（例：`wandb.init(job_type="analysis")`）。
- **既存 Run のアップデート**: W&B Run に紐づくデータをアップデートできます。例えば Run の config に、もともと記録していなかったアーキテクチャーやハイパーパラメーターなどの追加情報を加えることもできます。

利用可能な関数については [リファレンスドキュメント]({{< relref path="/ref/python/public-api/" lang="ja" >}}) を参照してください。

### API キーの作成

API キーはマシンを W&B に認証するために使用します。ユーザープロファイルから API キーを発行できます。

{{% alert %}}
より簡単な方法として、[W&B 認証ページ](https://wandb.ai/authorize) から直接 API キーを発行することもできます。表示された API キーは、パスワードマネージャーなど安全な場所に保管してください。
{{% /alert %}}

1. 右上のユーザープロファイルアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された API キーをコピーしてください。API キーを非表示にするにはページを再読み込みしてください。


### Run パスの確認

Public API を使う際、多くの場合 `<entity>/<project>/<run_id>` 形式の Run パスが必要になります。アプリの UI で Run ページを開き、[Overview タブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}}) をクリックすると Run パスを取得できます。


### Run データのエクスポート

完了済み、または進行中の Run からデータをダウンロードできます。主な利用例は、Jupyter notebook での独自分析用のデータフレーム取得や、自動化環境でのカスタムロジック利用です。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

Run オブジェクトでよく使われる属性は次の通りです。

| 属性         | 意味                                                                                                                                                                                                                                                                                                    |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`    | Run の設定情報（例：トレーニングのハイパーパラメータやデータセット作成時の前処理メソッドなど）を格納した辞書です。Run の "入力" 情報として捉えましょう。                                                                                                 |
| `run.history()` | モデルのトレーニング中などで変化する値（例えば loss など）を時系列で格納する辞書のリストです。`run.log()` コマンドでこのオブジェクトに値が追加されます。                                                                                 |
| `run.summary`   | Run の結果を要約した情報を持つ辞書です。精度や loss のようなスカラー値や大きなファイルも含みます。デフォルトでは、`run.log()` により最後に記録した値が summary となります。summary の内容は直接指定することもできます。Run の "出力" と考えてください。 |

過去 Run のデータを変更・更新することもできます。デフォルトでは、api オブジェクトのインスタンス1つが全ネットワークリクエストをキャッシュします。実行中のスクリプトでリアルタイムの値が必要な場合は、`api.flush()` を呼ぶことで最新値を取得できます。

### Run 属性の理解

以下のコードスニペットは、Run を作成しデータをログ、その後 Run 属性にアクセスする例です。

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

この Run オブジェクトでの属性ごとの出力例は以下の通りです。

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

デフォルトの history メソッドは、メトリクスを指定されたサンプル数（デフォルト500）にサンプリングします（`samples` 引数で変更可能）。大規模な Run の全データをエクスポートしたい場合は `run.scan_history()` を使います。詳細は [APIリファレンス]({{< relref path="/ref/python/public-api" lang="ja" >}}) をご覧ください。

### 複数 Runs のクエリ

{{< tabpane text=true >}}
    {{% tab header="DataFrame と CSV" %}}
この例では、プロジェクト内の Run を検索し、Run 名・config・summary を含む CSV を出力します。`<entity>` と `<project>` を自分の W&B entity/プロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には精度などの出力メトリクスが含まれます
    #  _json_dict で大きなファイルは省略します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメータが含まれます
    #  _ で始まる特殊値は省きます
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は Run の人間が読める名前です
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")

run.finish()
```
    {{% /tab %}}
    {{% tab header="MongoDB スタイル" %}}
W&B API では api.runs() でプロジェクト内の Run をクエリできます。主な用途は独自分析のための Run データのエクスポートです。クエリインターフェースは [MongoDB のもの](https://docs.mongodb.com/manual/reference/operator/query)と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}


`api.runs` を呼ぶと `Runs` オブジェクトが返され、これはイテラブルでリストのように扱えます。デフォルトでは順次50 Runずつ取得しますが、`per_page` キーワード引数で1ページ内の数を変更できます。

また、`order` キーワード引数で並び順も指定できます。デフォルトは `-created_at`。昇順にしたい場合は `+created_at` を指定してください。また、config や summary の値（例: `summary.val_acc` や `config.experiment_name`）でのソートも可能です。

### エラー処理

W&B サーバーとの通信でエラーが発生した場合は `wandb.CommError` が発生します。元の例外は `exc` 属性から調べられます。

### API で最新の git commit を取得

UI で Run の Overview タブを見ると最新の git commit が表示されます。これは `wandb-metadata.json` ファイルにも記録されます。Public API からは `run.commit` で git hash を取得できます。

### Run 実行時に名前や ID を取得

`wandb.init()` 実行後、スクリプト内からランダムな Run ID や人が読める Run 名を次のように参照できます：

- ユニークな Run ID（8文字ハッシュ）: `run.id`
- ランダムな Run 名（人が読める名前）: `run.name`

Run の識別名を工夫したい場合は以下がおすすめです：

- **Run ID**: 生成されたハッシュのままにする（プロジェクト内でユニークである必要があるため）。
- **Run 名**: 短く・読めて・できればユニークなものにする（チャート上で見分けがつきやすくなります）。
- **Run ノート**: Run の簡単な説明などに活用できます。`wandb.init(notes="説明文")` で設定できます。
- **Run タグ**: Run タグで動的に管理し、UI のフィルタで必要な Run だけ表示可能です。スクリプトからタグ付けも UI から編集も可能で、Runs テーブルや Run ページの Overview タブからも編集できます。詳しくは[こちら]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})をご参照ください。

## Public API の使用例

### matplotlib や seaborn で可視化するためデータをエクスポート

よくあるエクスポートパターンは[API例]({{< relref path="/ref/python/public-api/" lang="ja" >}})をご覧ください。カスタムプロットや展開された Runs テーブルのダウンロードボタンから直接 CSV を取得することも可能です。

### Run からメトリクスを読み取る

この例は、Run に保存された `accuracy` とそのタイムスタンプを出力します。`run.log({"accuracy": acc})` で保存した場合、`"<entity>/<project>/<run_id>"` で取得できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### Run のフィルタ

MongoDB クエリ言語を使いフィルタできます。

#### 日付

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### Run から特定メトリクスのみ取得

特定メトリクスだけ取得したい場合は `keys` 引数を利用します。`run.history()` のデフォルトのサンプル数は500です。指定したメトリクスが存在しないステップは DataFrame 上 NaN になります。`keys` 引数を指定すると、該当メトリクスを持つステップを多くサンプルします。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2つの Run を比較

`run1` と `run2` の間で異なる config パラメータを出力します。

```python
import pandas as pd
import wandb

api = wandb.Api()

# <entity>, <project>, <run_id> を適宜置き換えてください
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

### 終了済み Run のメトリクスを更新

過去の Run の accuracy を `0.9` にセットしたり、accuracy のヒストグラムに `numpy_array` をセットしたりできます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 完了済み Run のメトリクス名を変更

summary テーブル内のカラム名を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
カラム名の変更はテーブルにのみ反映されます。チャートでは従来通り元のメトリクス名で参照されます。
{{% /alert %}}



### 既存 Run の config を更新

config の一部を更新する例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース消費量を CSV ファイルにエクスポート

以下のスニペットでシステムリソース消費量を取得し、CSV に保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### 全てのサンプリング無しメトリクスデータ取得

history から取得するデータはデフォルトで500点サンプリングですが、`run.scan_history()` で記録されたすべてのデータポイントを取得できます。以下は loss の全データをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history からページネーションでデータ取得

もしメトリクス取得が遅い/タイムアウトが起きる場合は、`scan_history` のページサイズを縮小することで個々のリクエストのタイムアウト回避を試せます。デフォルトページサイズは500です。最適なサイズを試してみてください：

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内の全 Run のメトリクスを CSV にエクスポート

プロジェクト内の Run を取得し、Run 名・config・summaryを含む DataFrame・CSV を作成します。`<entity>` と `<project>` を適宜置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には出力メトリクス
    #  accuracy などが含まれます
    #  _json_dict で大きなファイル省略
    summary_list.append(run.summary._json_dict)

    # .configはハイパーパラメータ
    #  _ で始まる値は省きます
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .nameは人が読める名前
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### Run の開始時間を取得

Run の作成日時（開始時刻）を取得するスニペットです。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了済み Run にファイルをアップロード

指定したファイルを既存の完了済み Run にアップロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### Run からファイルをダウンロード

Run ID uxte44z7 の cifar プロジェクトで "model-best.h5" を検索しローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### Run の全ファイルをダウンロード

Run に紐づく全ファイルを検索してローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定 Sweep の全 Run を取得

スニペットは特定 Sweep に紐づく全 Run をダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### Sweep からベストな Run を取得

以下は指定 Sweep からベストな Run を取得する例です。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は Sweep 設定の `metric` パラメータで定義された最良の Run になります。

### Sweep で検証精度が最高のモデルファイルをダウンロード

model.h5 に保存されたモデルファイルのうち、validation accuracy 最大のものをダウンロードします。

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

### 指定拡張子のファイルをすべて削除

このスニペットで特定の拡張子を持つファイルを Run からすべて削除します。

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

このスニペットは、Run の全システムリソース消費メトリクスを DataFrame 化し、CSV に保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### summary メトリクスの更新

summary メトリクスは辞書で更新可能です。

```python
summary.update({"key": val})
```

### Run の実行コマンド取得

各 Run には、Run の Overview ページに起動コマンドが記録されています。これを API から取得するには:

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```