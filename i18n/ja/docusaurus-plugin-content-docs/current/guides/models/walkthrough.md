---
description: Weights & Biasesを使ってモデル管理を学ぶ
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# クイックスタート

<head>
  <title>モデル管理の使い方を解説</title>
</head>

![](/images/models/walkthrough.png)

この解説では、Weights & Biasesを使ってモデル管理を行う方法を学びます。プロダクションモデルワークフローのトラッキング、データ可視化、レポーティングを完全にサポートします。

1. **モデルバージョン管理**: モデルと学習済みパラメータのすべてのバージョンを保存・復元し、ユースケースや目的別にバージョンを整理します。トレーニングのメトリクスをトラッキングし、カスタムメタデータを追加し、モデルのリッチなマークダウン説明を文書化します。
2. **モデルの履歴:** モデルを生成するために使用された正確なコード、ハイパーパラメータ、およびトレーニングデータセットをトラッキングします。モデルの再現性を実現します。
3. **モデルライフサイクル:** 「ステージング」や「プロダクション」のような位置に有望なモデルを昇格させ、ダウンストリームのユーザーが最適なモデルを自動的に取得できるようにします。レポートでコラボレーションしながら進捗を共有します。

_私たちは現在、新しいモデル管理機能を積極的に開発しています。質問や提案がありましたら、support@wandb.comまでお気軽にお問い合わせください。_

:::info
モデルレジストリのすべての利用可能なコンテンツについては、[アーティファクトタブ](https://docs.wandb.ai/ref/app/pages/project-page#artifacts-tab)の詳細を参照してください！
:::

## ワークフロー

ここからは、トレーニング済みモデルの生成、整理、利用における標準的なワークフローについて解説します。
1. [新しいRegistered Modelを作成する](walkthrough.md#1.-create-a-new-model-portfolio)
2. [モデルバージョンのトレーニングとログ](walkthrough.md#2.-train-and-log-model-versions)
3. [Registered Modelにモデルバージョンをリンクする](walkthrough.md#3.-link-model-versions-to-the-portfolio)
4. [モデルバージョンの使用方法](walkthrough.md#4.-use-a-model-version)
5. [モデルパフォーマンスを評価する](walkthrough.md#5.-evaluate-model-performance)
6. [バージョンをプロダクションに昇格させる](walkthrough.md#6.-promote-a-version-to-production)
7. [プロダクションモデルを推論に利用する](walkthrough.md#7.-consume-the-production-model)
8. [レポートダッシュボードを作成する](walkthrough.md#8.-build-a-reporting-dashboard)

:::tip
**ステップ2〜3をカバーする最初のコードブロックと、ステップ4〜6をカバーする2つ目のコードブロックが含まれる** [**Colabノートブックが提供されています**](https://colab.research.google.com/drive/1wjgr9AHICOa3EM1Ikr_Ps_MAm5D7QnCC) **。**
:::

![](/images/models/workflow_dag.png)

### 1. 新しいRegistered Modelを作成する

まず、特定のモデリングタスクに対するすべての候補モデルを格納するRegistered Modelを作成します。このチュートリアルでは、古典的な[MNIST Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)（0-9の出力クラスを持つ28x28グレースケール入力画像）を使用します。以下のビデオでは、新しいRegistered Modelの作成方法を説明しています。

<Tabs
  defaultValue="registry"
  values={[
    {label: 'モデルレジストリを使用', value: 'registry'},
    {label: 'アーティファクトブラウザを使用', value: 'browser'},
    {label: 'プログラマティックリンク', value: 'programmatic'},
  ]}>
  <TabItem value="registry">

1. [wandb.ai/registry/model](https://wandb.ai/registry/model)（ホームページからリンクされている）で、モデルレジストリを開きます。
![](/images/models/create_registered_model_1.png)

![](/images/models/create_registered_model_2.png)

2. モデルレジストリの上部にある `Create Registered Model` ボタンをクリックします。

![](/images/models/create_registered_model_3.png)

3. `Owning Entity` と `Owning Project` が望む値に正しく設定されていることを確認してください。新しい登録済みモデルには、モデリングタスクや関心のあるユースケースを説明するユニークな名前を入力します。

![](/images/models/create_registered_model_4.png)
  </TabItem>
  <TabItem value="browser">

1. プロジェクトのアーティファクトブラウザにアクセスします: `wandb.ai/<entity>/<project>/artifacts`
2. アーティファクトブラウザのサイドバーの下部にある `+` アイコンをクリックします
3. `Type: model`, `Style: Collection`, および名前を選択します。この例では、`MNIST Grayscale 28x28` です。コレクションはモデリングタスクにマッピングする必要があります。ユースケースを説明するユニークな名前を入力してください。

![](/images/models/browser.gif)
  </TabItem>
    <TabItem value="programmatic">

すでにログされたモデルのバージョンがある場合、SDKから直接登録済みモデルにリンクできます。指定された登録済みモデルが存在しない場合、自動的に作成されます。

手動でのリンクは一度限りのモデルに便利ですが、モデルのバージョンをコレクションにプログラムでリンクすることがよくある場面でも便利です。例えば、毎晩のジョブやCIパイプラインが、すべてのトレーニングジョブから最良のモデルバージョンをリンクしたい場合です。コンテキストとユースケースに応じて、3つの異なるリンクAPIのいずれかを使用することがあります。

**Public APIからモデルアーティファクトを取得:**

```python
import wandb
# API経由でモデルバージョンを取得
art = wandb.Api().artifact(...)
# モデルバージョンをモデルコレクションにリンク
art.link("[[entity/]project/]collectionName")
```

**現在のRunではモデルアーティファクトが「使用されている」:**

```python
import wandb
# W&B runを初期化してトラッキングを開始
wandb.init()
# モデルバージョンの参照を取得
art = wandb.use_artifact(...)
# モデルバージョンをモデルコレクションにリンク
art.link("[[entity/]project/]collectionName")
```

**現在のRunではモデルアーティファクトがログされている：**

```python
import wandb
# W&B runを初期化してトラッキングを開始
wandb.init()
# モデルバージョンを作成
art = wandb.Artifact(...)
# モデルバージョンをログ
wandb.log_artifact(art)
# モデルバージョンをコレクションにリンク
wandb.run.link_artifact(art, "[[entity/]project/]collectionName")
```
  </TabItem>
</Tabs>
### 2. モデルのバージョンをトレーニング＆ログに残す

次に、トレーニングスクリプトからモデルをログに記録します。

1. （オプション）データセットを依存関係として宣言し、再現性と監査性のために追跡できるようにします
2. モデリングライブラリ（例：[PyTorch](https://pytorch.org/tutorials/beginner/saving\_loading\_models.html) や [Keras](https://www.tensorflow.org/guide/keras/save\_and\_serialize)）が提供するシリアル化プロセスを使って、定期的に（および/またはトレーニング終了時に）モデルをディスクに**シリアル化**します。
3. "model"というタイプのArtifactにモデルファイルを**追加**します
   * 注：`f'mnist-nn-{wandb.run.id}'`という名前を使用しています。必須ではありませんが、Run idで"下書き"のアーティファクトを名前空間化することで、整理された状態を維持することが望ましいです。
4. （オプション）トレーニング中のモデルのパフォーマンスに関連するトレーニングメトリクスをログに記録します。
   * 注：モデルバージョンをログに記録する直前にログに記録されたデータは、そのバージョンに自動的に関連付けられます。
5. モデルを**ログ**に記録します。
   * 注：複数のバージョンをログに記録している場合、以前のバージョンよりもパフォーマンスが向上したモデルバージョンに "best" のエイリアスを追加することが望ましいです。これにより、特にトレーニングの末尾で過学習が発生する場合に、最高のパフォーマンスを持つモデルを簡単に見つけることができます。

デフォルトでは、シリアル化されたモデルをログに記録するために、ネイティブのW&B Artifacts APIを使用する必要があります。ただし、このパターンは非常に一般的であるため、シリアル化、Artifactの作成、ログの記録を組み合わせた単一のメソッドが提供されています。詳細については、「(Beta) `log_model`を使用する」タブを参照してください。

<Tabs
  defaultValue="withartifacts"
  values={[
    {label: 'Artifactsを使用する', value: 'withartifacts'},
    {label: 'データセット依存関係を宣言する', value: 'datasetdependency'},
    {label: '[Beta] `log_model()`を使用する', value: 'logmodel'},
  ]}>
  <TabItem value="withartifacts">

```python
import wandb

# W&B runを常に初期化して追跡を開始する
wandb.init()
# （オプション）上流のデータセット依存関係を宣言する
# `Declare Dataset Dependency`タブを参照してください
# 代替の例については。
dataset = wandb.use_artifact("mnist:latest")

# 各エポックの終わりに（またはスクリプトの終わりに）...
# ... モデルをシリアル化します
model.save("path/to/model.pt")
# ... モデルバージョンを作成
art = wandb.Artifact(f'mnist-nn-{wandb.run.id}', type="model")
# ... シリアル化されたファイルを追加
art.add_file("path/to/model.pt", "model.pt")
# （オプション）トレーニングメトリクスをログに記録
wandb.log({"train_loss": 0.345, "val_loss": 0.456})
# ... バージョンをログに記録
if model_is_best:
    # モデルがこれまでの最高のモデルである場合、
    # エイリアスに "best" を追加
    wandb.log_artifact(art, aliases=["latest", "best"])
else:
    wandb.log_artifact(art)
```
  </TabItem>
  <TabItem value="datasetdependency">

トレーニングデータをトラッキングしたい場合は、データセットで `wandb.use_artifact` を呼び出すことで依存関係を宣言できます。以下は、データセット依存関係を宣言する方法の3つの例です。

**W&Bに保存されているデータセット**

``` python
dataset = wandb.use_artifact("[[entity/]project/]name:alias")
```
**ローカルファイルシステムに保存されたデータセット**

```python
art = wandb.Artifact("dataset_name", "dataset")
art.add_dir("path/to/data") # または art.add_file("path/to/data.csv")
dataset = wandb.use_artifact(art)
```

**リモートバケットに保存されたデータセット**

```python
art = wandb.Artifact("dataset_name", "dataset")
art.add_reference("s3://path/to/data")
dataset = wandb.use_artifact(art)
```
  </TabItem>
    <TabItem value="logmodel">

:::warning
以下のコードスニペットは、積極的に開発されている`beta` APIを利用しているため、変更される可能性があり、後方互換性が保証されていません。
:::

```python
from wandb.beta.workflows import log_model

# (オプション) 上流のデータセット依存関係を宣言する
# 別の例については、「Declare Dataset Dependency」タブを参照してください。
dataset = wandb.use_artifact("mnist:latest")
# （オプション）トレーニングメトリクスをログする
wandb.log({"train_loss": 0.345, "val_loss": 0.456})

# このメソッドは、モデルをシリアライズし、runを開始し、バージョンを作成し、
# ファイルをバージョンに追加し、バージョンをログする。デフォルトの名前、プロジェクト
# エイリアス、メタデータなどを上書きすることができます。
log_model(model, "mnist-nn", aliases=["best"] if model_is_best else [])
```

:::info
注：カスタムのシリアライズおよびデシリアライズの戦略を定義することができます。[`_SavedModel` クラス](https://github.com/wandb/wandb/blob/9dfa60b14599f2716ab94dd85aa0c1113cb5d073/wandb/sdk/data\_types/saved\_model.py#L73)をサブクラス化して、[`_PytorchSavedModel` クラス](https://github.com/wandb/wandb/blob/9dfa60b14599f2716ab94dd85aa0c1113cb5d073/wandb/sdk/data\_types/saved\_model.py#L358)と同様の方法で行えます。すべてのサブクラスは、シリアライズの登録に自動的にロードされます。これはベータ機能であるため、質問やコメントがあれば、support@wandb.comまでお問い合わせください。
:::
  </TabItem>
</Tabs>


1つまたは複数のモデルバージョンをログした後、Artifact Browserに新しいモデルアーティファクトが表示されることに気付くでしょう。ここでは、`mnist_nn-1r9jjogr`という名前のアーティファクトに5つのバージョンをログした結果を見ることができます。

![](/images/models/train_log_model_version_browser.png)

例のノートブックに沿っている場合、以下の画像に似たチャートが表示されるRun Workspaceが表示されるはずです。

![](/images/models/train_log_model_version_notebook.png)

### 3. モデルバージョンを登録モデルにリンクする

さて、モデルバージョンのどれかを登録モデルにリンクする準備ができたとしましょう。これは、手動でもAPI経由でも実行できます。

<Tabs
  defaultValue="manual_link"
  values={[
    {label: '手動でのリンク', value: 'manual_link'},
    {label: 'プログラムでのリンク', value: 'program_link'},
    {label: '[ベータ] `log_model()`を使用する', value: 'logmodel'},
  ]}>
  <TabItem value="manual_link">
以下のビデオは、モデルバージョンを新しく作成した登録済みモデルに手動でリンクする方法を説明しています。

1. 対象のモデルバージョンに移動する
2. リンクアイコンをクリックする
3. ターゲットとする登録済みモデルを選択する
4. （オプション）：追加のエイリアスを追加する

![](/images/models/link_model_versions.gif)
  </TabItem>
  <TabItem value="program_link">

手動でのリンクは1回限りのモデルに便利ですが、モデルバージョンをコレクションにプログラムでリンクすることがよくあります。例えば、毎晩のジョブやCI開発フローで、すべてのトレーニングジョブから最良のモデルバージョンをリンクしたい場合などです。コンテキストやユースケースに応じて、以下の3種類のリンクAPIのいずれかを使用することがあります。

**パブリックAPIからモデルアーティファクトを取得する：**

```python
import wandb

# APIを使ってモデルバージョンを取得する
art = wandb.Api().artifact(...)

# モデルバージョンをモデルコレクションにリンクする
art.link("[[entity/]project/]collectionName")
```

**現在のRunでモデルアーティファクトが「使用」される：**

```python
import wandb
以下のMarkdownテキストを日本語に翻訳してください。翻訳したテキストのみを返し、それ以外のことは言わないでください。テキスト：

# W&B runを開始してトラッキングを開始
wandb.init()

# モデルバージョンへの参照を取得
art = wandb.use_artifact(...)

# モデルバージョンをモデルコレクションにリンク
art.link("[[entity/]project/]collectionName")
```

**モデルアーティファクトは現在のRunによってログされます：**

```python
import wandb

# W&B runを開始してトラッキングを開始
wandb.init()

# モデルバージョンを作成
art = wandb.Artifact(...)

# モデルバージョンをログ
wandb.log_artifact(art)

# モデルバージョンをコレクションにリンク
wandb.run.link_artifact(art, "[[entity/]project/]collectionName")
```
  </TabItem>
  <TabItem value="logmodel">
:::warning
以下のコードスニペットは、積極的に開発されている`beta`APIを利用しているため、互換性がない変更が発生することがあります。
:::

上記で説明したbetaの`log_model`を使用してモデルをログに残した場合、それに対応するメソッド`link_model`を使用できます。

```python
from wandb.beta.workflows import log_model, link_model

# モデルバージョンを取得する
model_version = wb.log_model(model, "mnist_nn")

# モデルバージョンをリンクする
link_model(model_version, "[[entity/]project/]collectionName")
```
  </TabItem>
</Tabs>


モデルバージョンをリンクすると、登録されたモデルのバージョンとソースアーティファクトとの間に、相互にハイパーリンクが表示されます。

![](@site/static/images/models/train_log_model_version.png)

### 4. モデルバージョンの使用

さて、モデルを使って評価を行ったり、データセットに対して予測を行ったり、ライブプロダクション環境で使用する準備が整いました。モデルをログに記録するときと同様に、生のArtifact APIを使用するか、もっと指向性のあるbeta APIを使用するかを選ぶことができます。

<Tabs
  defaultValue="usingartifacts"
  values={[
    {label: 'アーティファクトを使用', value: 'usingartifacts'},
    {label: '[Beta] `use_model()`を使用', value: 'use_model'},
  ]}>
  <TabItem value="usingartifacts">
モデルバージョンを`use_artifact`メソッドを使ってロードすることができます。

```python
import wandb

# 常にW&B runを初期化してトラッキングを開始する
wandb.init()

# モデルバージョンファイルをダウンロードする
path = wandb.use_artifact("[[entity/]project/]collectionName:latest").download()

# メモリ内でモデルオブジェクトを再構築する:
# 以下の`make_model_from_data`は、ディスクからモデルをロードするための
# あなたのデシリアル化ロジックを表しています
model = make_model_from_data(path)
```
  </TabItem>
  <TabItem value="use_model">

:::warning
次のコードスニペットでは、活発に開発されている`beta`APIを利用しているため、変更される可能性があり、後方互換性が保証されていません。
:::

モデルファイルを直接操作し、デシリアル化を処理するのは難しいです - 特に、モデルをシリアライズしたのが自分でない場合。`log_model`と対になるように、`use_model`はモデルを自動的にデシリアル化して再構築し、使用することができます。

```python
from wandb.beta.workflows import use_model

model = use_model("[[entity/]project/]collectionName").model_obj()
```
  </TabItem>
</Tabs>
### 5. モデル性能の評価

多くのモデルをトレーニングした後、それらのモデルのパフォーマンスを評価したくなるでしょう。ほとんどの場合、モデルがトレーニング中にアクセスできるデータセットとは独立したテストデータセットとして機能するホールドアウトデータがあります。モデルのバージョンを評価するには、まず上記のステップ4を完了して、モデルをメモリにロードする必要があります。そして:

1. （オプション）評価データに対してデータ依存関係を宣言する
2. メトリクス、メディア、テーブル、評価に役立つその他のものを**ログ**する

```python
# ... 4からの続き

# (オプション) 上流の評価データセット依存関係を宣言する
dataset = wandb.use_artifact("mnist-evaluation:latest")

# ユースケースに応じてモデルを評価する
loss, accuracy, predictions = evaluate_model(model, dataset)

# メトリクス、画像、テーブル、評価に役立つデータをログに出力する。
wandb.log(
    {
        "loss": loss, "accuracy": accuracy, 
        "predictions": predictions
        })
```

ノートブックで示されているような類似のコードを実行している場合、以下の画像に似たワークスペースが表示されるはずです。ここでは、テストデータに対するモデルの予測まで表示しています！

![](/images/models/evaluate_model_performance.png)

### 6. バージョンをプロダクションに昇格させる
次に、登録済みモデルの中でどのバージョンをプロダクション用として使うかを示すことができます。この際、エイリアスの概念を利用します。各登録済みモデルは、ユースケースに適した任意のエイリアスを持つことができますが、一般的には `production` が最も一般的なエイリアスです。各エイリアスは、一度に1つのバージョンにしか割り当てられません。

<Tabs
  defaultValue="UI_interface"
  values={[
    {label: 'UIインターフェースとともに', value: 'UI_interface'},
    {label: 'APIとともに', value: 'api'},
  ]}>
  <TabItem value="UI_interface">

![](/images/models/promote_version_to_prod_1.png)
  </TabItem>
  <TabItem value="api">

[パート3. モデルバージョンをコレクションにリンクする](walkthrough.md#3.-linking-model-versions-to-the-portfolio)の手順に従い、`aliases` パラメータに追加したいエイリアスを追加してください。
  </TabItem>
</Tabs>


以下の画像は、登録済みモデルのv1に新しく追加された `production` エイリアスを示しています！

![](/images/models/promote_version_to_prod_2.png)

### 7. プロダクションモデルの利用

<!-- 最後に、プロダクションモデルを推論に使用したい場合があります。これを行うには、単に [パート4. モデルバージョンの使用](walkthrough.md#4.-evaluate-model-performance) で概説されている手順に従い、`production` エイリアスを使用します。例えば： -->

```python
wandb.use_artifact("[[entity/]project/]registeredModelName:production")
```
バージョンを登録済みモデル内で参照するには、異なるエイリアス戦略を使用できます。



* `latest` - 最も最近リンクされたバージョンを取得します

* `v#` - `v0`、`v1`、`v2`などを使用して、登録済みモデルの特定のバージョンを取得できます

* `production` - あなたとあなたのチームが割り当てた任意のカスタムエイリアスを使用できます



### 8. レポーティングダッシュボードの構築



Weaveパネルを使って、モデルレジストリ/アーティファクトのビューをレポート内に表示できます！[こちらのデモ](https://wandb.ai/timssweeney/model_management_docs_official_v0/reports/MNIST-Grayscale-28x28-Model-Dashboard--VmlldzoyMDI0Mzc1)をご覧ください。以下は、例のモデルダッシュボードの全画面スクリーンショットです。



![](/images/models/build_reporting_dashboard.png)