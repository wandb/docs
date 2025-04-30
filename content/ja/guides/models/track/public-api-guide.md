---
title: データのインポートとエクスポート
description: MLFlow から データ をインポートし、保存した データ を W&B にエクスポートまたは更新します。
menu:
  default:
    identifier: ja-guides-models-track-public-api-guide
    parent: experiments
weight: 8
---

データをエクスポートまたはインポートするには、W&B パブリックAPIを使用します。

{{% alert %}}
この機能には python>=3.8 が必要です
{{% /alert %}}

## MLFlow からデータをインポート

W&B は、MLFlow からのデータのインポートをサポートしており、実験、runs、アーティファクト、メトリクス、その他のメタデータを含みます。

依存関係をインストール：

```shell
# 注意: これは py38+ が必要です
pip install wandb[importers]
```

W&B にログインします。初めてログインする場合は、表示されるプロンプトに従ってください。

```shell
wandb login
```

既存の MLFlow サーバーからすべての run をインポートします:

```py
from wandb.apis.importers.mlflow import MlflowImporter

importer = MlflowImporter(mlflow_tracking_uri="...")

runs = importer.collect_runs()
importer.import_runs(runs)
```

デフォルトでは、`importer.collect_runs()` は MLFlow サーバーからすべての run を収集します。特定のサブセットをアップロードしたい場合は、自分で runs イテラブルを構築し、それをインポーターに渡すことができます。

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
Databricks MLFlow からインポートする場合、最初に[Databricks CLI を設定することが必要です](https://docs.databricks.com/dev-tools/cli/index.html)。

前のステップで `mlflow-tracking-uri="databricks"` を設定します。
{{% /alert %}}

アーティファクトのインポートをスキップするには、`artifacts=False` を渡します:

```py
importer.import_runs(runs, artifacts=False)
```

特定の W&B エンティティとプロジェクトにインポートするには、`Namespace` を渡します:

```py
from wandb.apis.importers import Namespace

importer.import_runs(runs, namespace=Namespace(entity, project))
```

## データのエクスポート

パブリックAPIを使用して、W&B に保存したデータをエクスポートまたは更新します。このAPIを使用する前に、スクリプトからデータをログします。詳細は[クイックスタート]({{< relref path="/guides/quickstart.md" lang="ja" >}})を確認してください。

**パブリックAPIのユースケース**

- **データのエクスポート**: カスタム分析のために Jupyter ノートブックにデータフレームを取り込みます。データを探索した後、例えば `wandb.init(job_type="analysis")` のように新しい分析 run を作成し、結果を記録して学びを同期できます。
- **既存の runs の更新**: W&B run に関連して記録されたデータを更新することができます。例えば、最初はログされていなかったアーキテクチャーやハイパーパラメーターの情報を追加するために設定を更新することがあるでしょう。

利用可能な関数の詳細については、[生成されたリファレンスドキュメント]({{< relref path="/ref/python/public-api/" lang="ja" >}})を参照してください。

### APIキーを作成する

APIキーは、マシンをW&Bに認証します。ユーザープロフィールからAPIキーを生成できます。

{{% alert %}}
より効率的なアプローチとして、直接 [https://wandb.ai/authorize](https://wandb.ai/authorize) にアクセスしてAPIキーを生成できます。表示されたAPIキーをコピーし、パスワードマネージャーのような安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **ユーザー設定** を選択し、**APIキー** セクションまでスクロールします。
3. **表示** をクリックします。表示されるAPIキーをコピーします。APIキーを非表示にするには、ページを再読み込みします。

### run パスを見つける

パブリックAPIを使用するには、しばしば `<entity>/<project>/<run_id>` という形式の run パスが必要になります。アプリUIでrunページを開き、[Overviewタブ]({{< relref path="/guides/models/track/runs/#overview-tab" lang="ja" >}})をクリックしてrunパスを取得します。

### run データをエクスポートする

完了済みまたはアクティブな run からデータをダウンロードします。一般的な用途には、カスタム分析のために Jupyter ノートブックにデータフレームをダウンロードしたり、カスタムロジックを使用して自動化された環境で利用したりすることが含まれます。

```python
import wandb

api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

run オブジェクトで最もよく使用される属性は次のとおりです:

| 属性               | 意味                                                                                                                                                                                                                                                                                      |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config`       | run の設定情報を保持する辞書。トレーニング run のハイパーパラメーターや、データセットアーティファクトを作成するためのrun に使用する前処理メソッドなどが含まれます。これらはrun の入力と考えてください。                                                                                      |
| `run.history()`    | モデルがトレーニングされている間に変化する値を保存することを意図した辞書のリスト。`wandb.log()` コマンドはこのオブジェクトに追記します。                                                                                                                                                         |
| `run.summary`      | run の結果を総括した情報の辞書。これには精度や損失のようなスカラー値や大きなファイルが含まれます。デフォルトでは、`wandb.log()`がログされた時系列の最終値をサマリーに設定します。サマリーの内容は直接設定することもできます。サマリーはrun の出力と考えてください。                               |

過去の runs データも変更したり更新したりすることができます。デフォルトでは、APIオブジェクトのインスタンス1つにつき、すべてのネットワークリクエストがキャッシュされます。実行中のスクリプトでリアルタイム情報が必要な場合、`api.flush()` を呼び出して更新された値を取得してください。

### 属性の理解

以下のrunに対して:

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

これは上記のrunオブジェクトの属性に対する異なる出力です。

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

デフォルトの history メソッドは、メトリクスを固定されたサンプル数（デフォルトは500）にサンプリングします。大規模なrunのすべてのデータをエクスポートしたい場合は、`run.scan_history()` メソッドを使用してください。詳細は[APIリファレンス]({{< relref path="/ref/python/public-api" lang="ja" >}})を参照してください。

### 複数の runs のクエリ

{{< tabpane text=true >}}
    {{% tab header="データフレームとCSV" %}}
このサンプルスクリプトはプロジェクトを検索し、CSVにrun の名前、設定、サマリーステータスを出力します。 `<entity>` と `<project>` をそれぞれあなたのW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には、精度のようなメトリクスの出力キー/値が含まれています。
    #  ._json_dict を呼び出して大きなファイルを省略します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれています。
    #  _ で始まる特別な値は削除します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は、run の人間が読み取り可能な名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```
    {{% /tab %}}
    {{% tab header="MongoDB スタイル" %}}
W&B API は、api.runs() を使用してプロジェクト内の runs を横断してクエリを実行する方法も提供しています。最も一般的なユースケースはカスタム分析のために runs データをエクスポートすることです。クエリインターフェースは [MongoDB が使用するもの](https://docs.mongodb.com/manual/reference/operator/query) と同じです。

```python
runs = api.runs(
    "username/project",
    {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]},
)
print(f"Found {len(runs)} runs")
```    
    {{% /tab %}}
{{< /tabpane >}}

`api.runs` を呼び出すと、反復可能でリストのように動作する `Runs` オブジェクトが返されます。デフォルトでは、オブジェクトは要求に応じて1回に50のrunを順番にロードしますが、`per_page` キーワード引数を使用してページごとにロードする数を変更できます。

`api.runs` は`order`キーワード引数も受け取ります。デフォルトの順序は `-created_at` です。昇順にするには `+created_at` を指定してください。設定やサマリーの値でソートすることもできます。例えば、`summary.val_acc` または `config.experiment_name` です。

### エラーハンドリング

W&B サーバーと話す際にエラーが発生すると、`wandb.CommError`が発生します。元の例外は `exc` 属性を通じて調査できます。

### API を通じて最新の git コミットを取得する

UIでは、runをクリックし、その run ページの Overview タブをクリックして最新の git コミットを見ることができます。それはまた、ファイル `wandb-metadata.json` の中にあります。パブリックAPIを使用して、`run.commit`を使用して git ハッシュを取得できます。

### run 中の run の名前とIDを取得する

`wandb.init()` を呼び出した後、スクリプトからランダム run IDまたは人間読み書き可能なrun名にアクセスできます。

- 一意の run ID（8文字のハッシュ）：`wandb.run.id`
- ランダム run 名前（人間読み書き可能）：`wandb.run.name`

run の識別子を効果的に設定する方法について考えている場合、以下を推奨します：

- **run ID**：生成されたハッシュのままにしておきます。これはプロジェクト内の run で一意である必要があります。
- **run 名前**：短く、読み書き可能で、おそらく一意であるべきです。そうすれば、チャートの異なる線間で違いをつけることができます。
- **run ノート**：run内で何をしているかを簡単に説明するのに最適です。`wandb.init(notes="your notes here")` で設定できます。
- **run タグ**：run タグで動的に追跡し、UIでフィルターを使用して興味のある run に絞り込みます。スクリプトからタグを設定し、runsテーブルやrunページのoverviewタブでUIからも編集できます。詳細な指示は[こちら]({{< relref path="/guides/models/track/runs/tags.md" lang="ja" >}})を参照してください。

## 公開APIの例

### matplotlib または seaborn で視覚化するためにデータをエクスポート

一般的なエクスポートパターンについては、[APIの例]({{< relref path="/ref/python/public-api/" lang="ja" >}})を確認してください。また、カスタムプロットや拡張されたrunテーブルでダウンロードボタンをクリックして、ブラウザからCSVをダウンロードすることもできます。

### run からメトリクスを読む

この例では、`wandb.log({"accuracy": acc})`で保存されたrunのタイムスタンプと精度を出力します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history().iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### runs のフィルタリング

MongoDBクエリ言語を使用してフィルターできます。

#### 日付

```python
runs = api.runs(
    "<entity>/<project>",
    {"$and": [{"created_at": {"$lt": "YYYY-MM-DDT##", "$gt": "YYYY-MM-DDT##"}}]},
)
```

### 特定の run のメトリクスを読む

run から特定のメトリクスを取り出すには、`keys` 引数を使用します。`run.history()` の場合、デフォルトのサンプル数は500です。特定のメトリクスを含まないログステップは、出力データフレームで `NaN` として表示されます。`keys` 引数を使用すると、APIは指定したメトリックキーを含むステップをより頻繁にサンプリングします。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
    for i, row in run.history(keys=["accuracy"]).iterrows():
        print(row["_timestamp"], row["accuracy"])
```

### 2つの run を比較する

これは `run1` と `run2` の間で異なる設定パラメーターを出力します。

```python
import pandas as pd
import wandb

api = wandb.Api()

# `<entity>`, `<project>`, `<run_id>` で置き換える
run1 = api.run("<entity>/<project>/<run_id>")
run2 = api.run("<entity>/<project>/<run_id>")


df = pd.DataFrame([run1.config, run2.config]).transpose()

df.columns = [run1.name, run2.name]
print(df[df[run1.name] != df[run2.name]])
```

出力：

```
              c_10_sgd_0.025_0.01_long_switch base_adam_4_conv_2fc
batch_size                                 32                   16
n_conv_layers                               5                    4
optimizer                             rmsprop                 adam
```

### run が完了した後に、run のメトリクスを更新する

この例では、以前のrunの精度を `0.9` に設定します。また、numpy_array のヒストグラムに以前のrunの精度ヒストグラムを変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```

### 完了した run でメトリクスをリネームする

この例ではテーブル内のサマリー列の名前を変更します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["new_name"] = run.summary["old_name"]
del run.summary["old_name"]
run.summary.update()
```

{{% alert %}}
列のリネームはテーブルにのみ適用されます。チャートは元の名前でメトリクスを参照し続けます。
{{% /alert %}}

### 既存の run の設定を更新する

この例では設定のひとつを更新します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソースの消費をCSVファイルにエクスポートする

以下のスニペットは、システムリソースの消費を見つけ、それらをCSVに保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サンプリングされていないメトリクスデータを取得する

history からデータを取得するとき、デフォルトでは500ポイントにサンプリングされます。`run.scan_history()`を使用すると全てのログデータポイントが取得できます。以下は、historyでログされたすべての `loss` データポイントをダウンロードする例です。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### history からページ分割されたデータを取得する

メトリクスがバックエンドでゆっくりと取得されている場合やAPIリクエストがタイムアウトしている場合、scan_history でページサイズを下げて、個々のリクエストがタイムアウトしないようにすることができます。デフォルトのページサイズは500なので、どのサイズが最適か試してみてください。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内のすべてのrun からメトリクスをCSVファイルにエクスポートする

このスクリプトは、プロジェクト内のrunを取得し、その名前、設定、およびサマリーステータスを含むデータフレームとCSVを生成します。 `<entity>` と `<project>` をそれぞれあなたのW&Bエンティティとプロジェクト名に置き換えます。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary には、精度のようなメトリクスの出力キー/値が含まれています。
    #  ._json_dict を呼び出して大きなファイルを省略します
    summary_list.append(run.summary._json_dict)

    # .config にはハイパーパラメーターが含まれています。
    #  _ で始まる特別な値は削除します。
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name は、run の人間が読み取り可能な名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

runs_df.to_csv("project.csv")
```

### run の開始時間を取得する

このコードスニペットは、run が作成された時間を取得します。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 完了した run にファイルをアップロードする

以下のコードスニペットは、選択したファイルを完了したrunにアップロードします。

```python
import wandb

api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```

### run からファイルをダウンロードする

これは、cifar プロジェクトの run ID uxte44z7 に関連付けられたファイル "model-best.h5" を見つけ、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### run からすべてのファイルをダウンロードする

これはrunに関連付けられたすべてのファイルを見つけ、ローカルに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定のスイープから run を取得する

このスニペットは特定のスイープに関連するすべてのrunをダウンロードします。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### スイープから最高の run を取得する

次のスニペットは、与えられたスイープから最高のrun を取得します。

```python
import wandb

api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run` は、スイープの設定の `metric` パラメータで定義されたメトリクスが最高のrunです。

### スイープから最高のモデルファイルをダウンロードする

このスニペットは、runで`model.h5`にモデルファイルを保存したスイープから、最高の検証精度を持つモデルファイルをダウンロードします。

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

### run から特定の拡張子のすべてのファイルを削除する

このスニペットは、runの特定の拡張子を持つファイルを削除します。

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

このスニペットは、run のすべてのシステムリソース消費メトリクスのデータフレームを生成し、CSVに保存します。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サマリーメトリクスを更新する

サマリーメトリクスを更新する辞書を渡すことができます。

```python
summary.update({"key": val})
```

### run を実行したコマンドを取得する

各runは、runの概要ページでそれを開始したコマンドをキャプチャします。このコマンドを API から取得するには次のように実行できます。

```python
import wandb

api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```