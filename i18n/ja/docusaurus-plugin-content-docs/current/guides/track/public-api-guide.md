---
description: Import data from MLFlow, export or update data that you have saved to W&B
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# データのインポートとエクスポート

<head>
  <title>W&Bへのデータのインポートとエクスポート</title>
</head>

W&B Public APIsを使用して、MLFlowからデータをインポートまたはエクスポートします。

## MLFLowからデータをインポートする

W&Bは、MLFlowからのデータのインポートをサポートしており、実験、runs、アーティファクト、メトリクス、その他のメタデータなどが含まれます。

#### クイックスタート
依存関係をインストール：
```sh
pip install mlflow wandb>=0.14.0
```

W&Bにログイン（まだログインしていない場合はプロンプトに従います）
```sh
wandb login
```

既存のMLFlowサーバーからすべてのrunsをインポートする：
```sh
wandb import mlflow \ &&
    --mlflow-tracking-uri <mlflow_uri> \ &&
    --target_entity       <entity> \ &&
    --target_project      <project>
```

#### 高度な使い方
Pythonからインポートすることもできます。これは、オーバーライドを指定したり、コマンドラインよりもPythonを好む場合に便利です。
```py
from wandb.apis.importers import MlflowImporter


# すべてのインポートされたrunsの設定を上書きするためのオプションのdict
overrides = {
    "entity": "my_custom_entity",
    "project": "my_custom_project"
}

importer = MlflowImporter(mlflow_tracking_uri="...")
importer.import_all_parallel()
```

さらに詳細な制御を行いたい場合は、実験を選択的にインポートしたり、独自のカスタムロジックに基づいて上書き設定を指定したりできます。たとえば、次のコードは、カスタムタグが付いたrunsを作成し、指定されたプロジェクトにインポートする方法を示しています。
```py
default_settings = {
    "entity": "default_entity",
    "project": "default_project"
}

special_tag_settings = {
    "entity": "special_entity",
    "project": "special_project"
}

for run in importer.download_all_runs():
    if "special_tag" in run.tags():
        overrides = special_tag_settings
    else:
        overrides = default_settings

    importer.import_run(run, overrides=overrides)
```


## データのエクスポート

Public APIを使用して、W&Bに保存したデータをエクスポートまたは更新します。このAPIを使用する前に、スクリプトからデータをログに記録する必要があります。詳細については、[クイックスタート](../../quickstart.md)をご確認ください。

**Public APIのユースケース**

* **データのエクスポート**：カスタム分析用のデータフレームをJupyterノートブックにダウンロードします。データを調べ終えたら、新しい分析runを作成して結果をログに記録することで、調査結果を同期できます。例：`wandb.init(job_type="analysis")`
* **既存のRunsの更新**：W&B runに関連付けられたデータを更新できます。たとえば、最初にログに記録されなかったアーキテクチャーやハイパーパラメーターなどの追加情報を含むように、一連のrunsのconfigを更新することができます。
[生成されたリファレンスドキュメント](https://docs.wandb.ai/ref/python/public-api)で利用可能な関数の詳細をご覧ください。

### 認証

お使いのマシンを[APIキー](https://wandb.ai/authorize)で2つの方法のいずれかで認証します。

1. コマンドラインで `wandb login` を実行し、APIキーを貼り付けます。
2. `WANDB_API_KEY` 環境変数にAPIキーを設定します。

### ランパスの取得

Public APIを使用するには、よく `<entity>/<project>/<run_id>` 形式のランパスが必要です。アプリUIで、ランページを開いて[概要タブ](../app/pages/run-page.md#overview-tab)をクリックしてランパスを取得します。

### ランデータのエクスポート

終了したランやアクティブなランからデータをダウンロードします。一般的な使用例には、Jupyter ノートブックでカスタム分析を行うためのデータフレームのダウンロードや、自動化された環境でのカスタムロジックの使用が含まれます。

```python
import wandb
api = wandb.Api()
run = api.run("<entity>/<project>/<run_id>")
```

ランオブジェクトの最も一般的に使用される属性は以下の通りです。

| 属性 | 意味 |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run.config` | ランの構成情報（トレーニングランのハイパーパラメーターや、データセットアーティファクトを作成するランの前処理方法など）の辞書です。これらはランの「入力」と考えてください。|
| `run.history()` | モデルのトレーニング中に変更される値（損失など）を格納するための辞書のリストです。`wandb.log()`コマンドはこのオブジェクトに追加します。 |
| `run.summary`   | ランの結果をまとめた情報の辞書です。これには、精度や損失などのスカラー値や大きなファイルが含まれます。デフォルトでは、`wandb.log()` は、ログされた時系列の最終値を要約に設定します。要約の内容は直接設定することもできます。要約をランの「出力」と考えてください。|

また、過去のランのデータを変更または更新することもできます。デフォルトでは、1つのAPIオブジェクトのインスタンスは、すべてのネットワークリクエストをキャッシュします。実行中のスクリプトでリアルタイム情報が必要なユースケースの場合は、`api.flush()`を呼び出して更新された値を取得します。

### さまざまな属性の理解

以下のランについて

```python
n_epochs = 5
config = {"n_epochs": n_epochs}
run = wandb.init(project=project, config=config)
for n in range(run.config.get("n_epochs")):
    run.log({"val": random.randint(0,1000), "loss": (random.randint(0,1000)/1000.00)})
run.finish()
```
これらは、上記のrunオブジェクト属性に対する異なる出力です

#### `run.config`

```python
{'n_epochs': 5}
```

#### `run.history()`

```python
   _step  val   loss  _runtime  _timestamp
0      0  500  0.244         4  1644345412
1      1   45  0.521         4  1644345412
2      2  240  0.785         4  1644345412
3      3   31  0.305         4  1644345412
4      4  525  0.041         4  1644345412
```

#### `run.summary`

```python
{'_runtime': 4,
 '_step': 4,
 '_timestamp': 1644345412,
 '_wandb': {'runtime': 3},
 'loss': 0.041,
 'val': 525}
```

### サンプリング

デフォルトの履歴メソッドは、メトリクスを固定されたサンプル数（デフォルトは 500、`samples`引数で変更できます）にサンプリングします。大規模なrunのすべてのデータをエクスポートしたい場合は、`run.scan_history()`メソッドを使用できます。詳細は [APIリファレンス](https://docs.wandb.ai/ref/python/public-api) を参照してください。

### 複数のrunsをクエリする

<Tabs
  defaultValue="dataframes_csvs"
  values={[
    {label: 'Dataframes and CSVs', value: 'dataframes_csvs'},
    {label: 'MongoDB Style', value: 'mongoDB'},
  ]}>
  <TabItem value="dataframes_csvs">

  
この例のスクリプトは、プロジェクトを検索し、名前、設定、およびサマリースタッツを含むCSVファイルを出力します。`<entity>`と`<project>`をそれぞれあなたのW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summaryには、精度などのメトリクスに関する
    # 出力キー/値が含まれます。
    # 大きなファイルを省略するために、._json_dictを呼び出します
    summary_list.append(run.summary._json_dict)

    # .configにはハイパーパラメーターが含まれます。
    # '_'で始まる特別な値を削除します。
    config_list.append(
        {k: v for k, v in run.config.items()
         if not k.startswith('_')})

    # .nameは、runの人間が読める名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")
```

  </TabItem>
  <TabItem value="mongoDB">

W&B APIはまた、api.runs()を使用してプロジェクト内の複数のrunを横断して問い合わせる方法も提供しています。最も一般的なユースケースは、カスタム分析のためにrunデータをエクスポートすることです。クエリインターフェイスは、[MongoDBが使用するもの](https://docs.mongodb.com/manual/reference/operator/query)と同じです。

```python
runs = api.runs("username/project",
    {"$or": [
        {"config.experiment_name": "foo"},
        {"config.experiment_name": "bar"}]
    })
print(f"Found {len(runs)} runs")
```

  </TabItem>
</Tabs>

`api.runs`を呼び出すと、`Runs`オブジェクトが返されます。このオブジェクトは、イテレーション可能で、リストとして機能します。デフォルトでは、オブジェクトは必要に応じて順番に50個のrunsを一度に読み込みますが、`per_page`キーワード引数でページあたりに読み込まれる数を変更することができます。

`api.runs`は、`order`キーワード引数も受け入れます。デフォルトの順序は`-created_at`で、`+created_at`を指定して昇順で結果を取得します。また、`summary.val_acc`や`config.experiment_name`などのconfigやsummaryの値でソートすることもできます。

### エラーハンドリング

W&Bサーバーと通信中にエラーが発生した場合、`wandb.CommError`が発生します。元の例外は`exc`属性を経由して調べることができます。

### APIを通じて最新のgitコミットを取得する

UIで、runをクリックしてからrunページの概要タブをクリックすると、最新のgitコミットが表示されます。また、`wandb-metadata.json`ファイルにもあります。パブリックAPIを使用して、`run.commit`でgitハッシュを取得できます。

## よくある質問

### matplotlibやseabornで可視化するためにデータをエクスポートする方法は？

[APIの例](https://docs.wandb.ai/library/public-api-guide#public-api-examples)で、一般的なエクスポートパターンをいくつかチェックしてください。さらに、カスタムプロットや拡張されたrunsテーブルのダウンロードボタンをクリックして、ブラウザからCSVをダウンロードすることもできます。

### run中にrunの名前とIDを取得する方法は？

`wandb.init()`を呼び出した後、スクリプトで以下のようにしてランダムなrun IDや人間が読めるrun名にアクセスできます。

* 一意のrun ID（8文字のハッシュ）：`wandb.run.id`
* ランダムなrun名（人間が読める）：`wandb.run.name`

runsに有用な識別子を設定する方法を考えている場合、以下がお勧めです。

* **Run ID**：生成されたハッシュのままにしておく。これは、プロジェクト内のruns間で一意である必要があります。
* **Run名**：これは、チャート上の異なる線を区別できるように、短く、読みやすく、できれば一意なものにするべきです。
* **Runのノート**：これは、runで行っている作業の簡単な説明を入れるのに適した場所です。`wandb.init(notes="ここにノートを入れてください")`で設定できます。
* **Runのタグ**：Runのタグで動的に情報を追跡し、UIのフィルターを使用して、気になるrunsだけの表に絞り込むことができます。スクリプトからタグを設定して、UIで編集できます。詳細な手順は[こちら](../app/features/tags.md)を参照してください。

## Public APIの例

### runからメトリクスを読み取る

この例では、`"<entity>/<project>/<run_id>"`に保存されたrunから、`wandb.log({"accuracy": acc})`で保存されたタイムスタンプと精度を出力します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
   for i, row in run.history().iterrows():
      print(row["_timestamp"], row["accuracy"])
```

### ランを絞り込む

MongoDBのクエリ言語を使用して、ランをフィルタリングできます。

#### 日付

```python
runs = api.runs('<entity>/<project>', {
    "$and": [{
    'created_at': {
        "$lt": 'YYYY-MM-DDT##',
        "$gt": 'YYYY-MM-DDT##'
        }
    }]
})
```

### ランから特定のメトリクスを取得する

ランから特定のメトリクスを取得するには、`keys` 引数を使用します。`run.history()` を使用する際のデフォルトのサンプル数は500です。特定のメトリクスが含まれていないログされたステップは、出力データフレームに `NaN` として表示されます。 `keys`引数を使用すると、リストされたメトリックキーを含むステップがより頻繁にサンプリングされます。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
if run.state == "finished":
   for i, row in run.history(keys=["accuracy"]).iterrows():
      print(row["_timestamp"], row["accuracy"])
```

### 2つのランを比較する

これにより、`run1` と `run2`の間で異なる設定パラメータが出力されます。
```python
import pandas as pd
import wandb
api = wandb.Api()

# あなたの<entity>、<project>、<run_id>に置き換えてください
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

### ランが終了した後に、ランのメトリクスを更新する

この例では、以前のランの精度を `0.9` に設定しています。また、以前のランの精度ヒストグラムを `numpy_array` のヒストグラムに変更しています。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary["accuracy"] = 0.9
run.summary["accuracy_histogram"] = wandb.Histogram(numpy_array)
run.summary.update()
```
### ランが終了した後に、メトリクスの名前を変更する

この例では、テーブルのサマリーカラムの名前を変更します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.summary['new_name'] = run.summary['old_name']
del run.summary['old_name']
run.summary.update()
```

:::caution
カラム名の変更はテーブルにのみ適用されます。チャートでは、元の名前でメトリクスが参照されます。
:::

### 既存のランの設定を更新する

この例では、設定のうち1つを更新します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.config["key"] = updated_value
run.update()
```

### システムリソース消費量をCSVファイルにエクスポートする

以下のスニペットでは、システムリソース消費量を検索し、それらをCSVに保存します。

```python
import wandb

run = wandb.Api().run("<entity>/<project>/<run_id>")

system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```
### サンプリングされていないメトリクスデータを取得

履歴からデータを取得すると、デフォルトで500ポイントにサンプリングされます。`run.scan_history()`を使用して、ログされたすべてのデータポイントを取得します。以下は、履歴にログされたすべての`loss`データポイントをダウンロードする例です。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
history = run.scan_history()
losses = [row["loss"] for row in history]
```

### 履歴からページネーションされたデータを取得

バックエンドでメトリックが遅く取得されている場合やAPIリクエストがタイムアウトしている場合は、`scan_history`でページサイズを下げて個々のリクエストがタイムアウトしないように試みることができます。デフォルトのページサイズは 500 ですので、最適なサイズを見つけるために異なるサイズで試してみてください:

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.scan_history(keys=sorted(cols), page_size=100)
```

### プロジェクト内のすべてのrunsからメトリクスをCSVファイルにエクスポート

このスクリプトは、プロジェクト内のrunsを取得し、runsの名前、設定、およびサマリーステータスを含むデータフレームとCSVを作成します。`<entity>`および`<project>`を、それぞれあなたのW&Bエンティティとプロジェクト名に置き換えてください。

```python
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "<entity>", "<project>"
runs = api.runs(entity + "/" + project)

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary は正確性のようなメトリックスの
    #  出力キー/バリューを含みます
    #  ._'json_dict' を呼ぶことで大きなファイルが省略されます
    summary_list.append(run.summary._json_dict)


# .configはハイパーパラメータを含んでいます。
    # ここでは、_で始まる特別な値を削除します。
    config_list.append(
        {k: v for k,v in run.config.items()
         if not k.startswith('_')})

    # .nameはrunの人間が読める名前です。
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    })

runs_df.to_csv("project.csv")
```

### Runの開始時間を取得する

このコードスニペットは、Runが作成された時間を取得します。

```python
import wandb
api = wandb.Api()

run = api.run("entity/project/run_id")
start_time = run.created_at
```

### 終了したRunにファイルをアップロードする

以下のコードスニペットは、選択したファイルを終了したRunにアップロードします。

```python
import wandb
api = wandb.Api()

run = api.run("entity/project/run_id")
run.upload_file("file_name.extension")
```
### runからファイルをダウンロードする

これは、cifarプロジェクトのrun ID uxte44z7に関連付けられた "model-best.h5" ファイルを見つけ、ローカルに保存します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
run.file("model-best.h5").download()
```

### runから全てのファイルをダウンロードする

これは、runに関連付けられたすべてのファイルを見つけ、ローカルに保存します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
for file in run.files():
    file.download()
```

### 特定のスイープからrunsを取得する

このスニペットは、特定のスイープに関連付けられたすべてのrunsをダウンロードします。

```python
import wandb
api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
sweep_runs = sweep.runs
```

### スイープから最良のrunを取得する

次のスニペットは、与えられたスイープから最良のrunを取得します。
```python
import wandb
api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
best_run = sweep.best_run()
```

`best_run`は、スイープ構成の`metric`パラメータで定義された最良の指標を持つrunです。

### スイープから最良のモデルファイルをダウンロードする

このスニペットは、`model.h5`にモデルファイルを保存したrunsのあるスイープから、最も高い検証精度を持つモデルファイルをダウンロードします。

```python
import wandb
api = wandb.Api()

sweep = api.sweep("<entity>/<project>/<sweep_id>")
runs = sorted(sweep.runs,
    key=lambda run: run.summary.get("val_acc", 0), reverse=True)
val_acc = runs[0].summary.get("val_acc", 0)
print(f"Best run {runs[0].name} with {val_acc}% val accuracy")

runs[0].file("model.h5").download(replace=True)
print("Best model saved to model-best.h5")
```

### runから特定の拡張子を持つすべてのファイルを削除する

このスニペットは、runから特定の拡張子を持つファイルを削除します。

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

このスニペットは、runのすべてのシステムリソース消費メトリクスを含むデータフレームを作成し、それをCSVに保存します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")
system_metrics = run.history(stream="events")
system_metrics.to_csv("sys_metrics.csv")
```

### サマリーメトリクスの更新

サマリーメトリクスを更新するために、ディクショナリを渡すことができます。

```python
summary.update({"key": val})
```

### runを実行したコマンドを取得する

各runは、runの概要ページでそれを起動したコマンドをキャプチャします。このコマンドをAPIから取得するには、次のように実行します。

```python
import wandb
api = wandb.Api()

run = api.run("<entity>/<project>/<run_id>")

meta = json.load(run.file("wandb-metadata.json").download())
program = ["python"] + [meta["program"]] + meta["args"]
```