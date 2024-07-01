---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 任意のライブラリに wandb を追加

このガイドでは、Pythonライブラリに W&B を統合して、強力な実験管理、GPUおよびシステム監視、モデルチェックポイントなどを取得するためのベストプラクティスを提供します。

:::note
まだ W&B の使い方を学んでいる場合は、このドキュメントの他の W&B ガイド（[実験管理](https://docs.wandb.ai/guides/track)など）を先に読むことをお勧めします。
:::

ここでは、作業中のコードベースが単一のPythonトレーニングスクリプトやJupyterノートブックよりも複雑な場合のベストティップとベストプラクティスを取り上げます。取り上げるトピックは次のとおりです:

* セットアップ要件
* ユーザーログイン
* wandb Run の開始
* Run Config の定義
* W&B へのログ
* 分散トレーニング
* モデルのチェックポイントとその他
* ハイパーパラメータチューニング
* 高度なインテグレーション

### セットアップ要件

開始する前に、ライブラリの依存関係として W&B を必須にするかどうかを決定してください:

#### インストール時に W&B を必須にする

依存関係ファイルに W&B Python ライブラリ（`wandb`）を追加します。例えば、`requirements.txt` ファイルに追加します。

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK (`wandb`) をオプションにする方法は2つあります:

A. ユーザーが `wandb` 機能を手動でインストールせずに使用しようとしたときにエラーを発生させ、適切なエラーメッセージを表示させます:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "You are trying to use wandb which is not currently installed"
        "Please install it using pip install wandb"
    ) 
```

B. Pythonパッケージを作成している場合、`pyproject.toml` ファイルに `wandb` をオプションの依存関係として追加します。

```toml
[project]
name = "my_awesome_lib"
version = "0.1.0"
dependencies = [
    "torch",
    "sklearn"
]

[project.optional-dependencies]
dev = [
    "wandb"
]
```

### ユーザーログイン

ユーザーがW&Bにログインする方法はいくつかあります:

<Tabs
  defaultValue="bash"
  values={[
    {label: 'Bash', value: 'bash'},
    {label: 'Notebook', value: 'notebook'},
    {label: 'Environment Variable', value: 'environment'},
  ]}>
  <TabItem value="bash">
ターミナルでBashコマンドを使用してW&Bにログイン

```bash
wandb login $MY_WANDB_KEY
```
  </TabItem>
  <TabItem value="notebook">
JupyterまたはColabノートブックの場合、次のようにしてW&Bにログイン

```python
import wandb
wandb.login()
```
  </TabItem>
  <TabItem value="environment">

APIキーのための[W&B環境変数](../track/environment-variables.md)を設定

```bash
export WANDB_API_KEY=$YOUR_API_KEY
```

または

```
os.environ['WANDB_API_KEY'] = "abc123..."
```
  </TabItem>
</Tabs>

ユーザーが上記の手順に従わずに初めてwandbを使用する場合、スクリプトで`wandb.init`が呼び出されると自動的にログインを促されます。

### wandb Run の開始

W&B Runは、W&Bによって記録される計算の単位です。通常、各トレーニング実験に対して1つのW&B Runを関連付けます。

次のコードを使ってW&Bを初期化してRunを開始します：

```python
wandb.init()
```

オプションとして、プロジェクトの名前を提供するか、`wandb_project` などのパラメータを使ってユーザー自身で設定させることができます。また、エンティティパラメータにはユーザー名またはチーム名を`wandb_entity`として設定します。

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init` を置く場所

あなたのライブラリでは、できるだけ早い段階で W&B Run を作成する必要があります。これは、コンソールに表示されるすべての出力（エラーメッセージを含む）が W&B Run の一部として記録されるため、デバッグが容易になるからです。

#### `wandb` をオプションとしてライブラリを実行

ユーザーがライブラリを使用するときに `wandb` をオプションにする場合、次のいずれかの方法を採用できます:

* `wandb` フラグを定義:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="python">

```python
trainer = my_trainer(..., use_wandb=True)
```
  </TabItem>
  <TabItem value="bash">

```bash
python train.py ... --use-wandb
```
  </TabItem>
</Tabs>

* または、`wandb.init` で `wandb` を無効に設定:

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="python">

```python
wandb.init(mode="disabled")
```
  </TabItem>
  <TabItem value="bash">

```bash
export WANDB_MODE=disabled
```
または

```bash
wandb disabled
```
  </TabItem>
</Tabs>

* または、`wandb` をオフラインに設定。ただし、これでも `wandb` は実行されますが、インターネット経由でW&Bに通信しようとしません。

<Tabs
  defaultValue="environment"
  values={[
    {label: 'Environment Variable', value: 'environment'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="environment">

```bash
export WANDB_MODE=offline
```

または

```python
os.environ['WANDB_MODE'] = 'offline'
```
  </TabItem>
  <TabItem value="bash">

```
wandb offline
```
  </TabItem>
</Tabs>

### wandb Run Config の定義

`wandb` run configを使用して、W&B Runの作成時にモデル、データセットなどに関するメタデータを提供できます。この情報を使用して、異なる実験を比較し、主要な違いをすばやく理解することができます。

![W&B Runs テーブル](/images/integrations/integrations_add_any_lib_runs_page.png)

記録できる典型的な config パラメータには次のようなものがあります:

* モデル名、バージョン、アーキテクチャパラメータなど
* データセット名、バージョン、訓練/検証例の数など
* トレーニングパラメータ（学習率、バッチサイズ、オプティマイザーなど）

次のコードスニペットはconfigの記録方法を示します:

```python
config = {"batch_size":32, …}
wandb.init(…, config=config)
```

#### wandb config の更新

`wandb.config.update`を使用してconfigを更新します。定義後にパラメータが取得される場合（例えば、モデルのインスタンス化後にモデルのパラメータを追加したい場合など）に、設定辞書を更新するのに役立ちます。

```python
wandb.config.update({"model_parameters": 3500})
```

configファイルの定義方法についての詳細は、[wandb.config で実験を設定](https://docs.wandb.ai/guides/track/config) を参照してください。

### W&Bへのログ

#### メトリクスのログ

キーがメトリクス名である辞書を作成し、この辞書オブジェクトを[`wandb.log`](https://docs.wandb.ai/guides/track/log)に渡します:

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        wandb.log(metrics)
```

多くのメトリクスがある場合、メトリクス名にプレフィックスを使用してUIで自動的にグループ化できます。例えば`train/...`と`val/...`を使用すると、トレーニングと検証メトリクスまたは別のメトリクスタイプのためのW&B Workspaceの別セクションを作成できます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
wandb.log(metrics)
```

![2つの別々のセクションがあるW&B Workspace](/images/integrations/integrations_add_any_lib_log.png)

`wandb.log`の詳細については、[wandb.logでデータをログ](https://docs.wandb.ai/guides/track/log)を参照してください。

#### x軸の不整合防止

同じトレーニングステップで複数回`wandb.log`を呼び出す必要がある場合があります。wandb SDKには内部ステップカウンタがあり、各`wandb.log`呼び出しごとにカウントが増加します。これにより、wandbログカウンタがトレーニングループ内のステップと整合しない可能性があります。

以下の例の最初のパスでは、`train/loss`の内部`wandb`ステップは0ですが、`eval/loss`の内部`wandb`ステップは1になります。次のパスでは、`train/loss`は2になり、`eval/loss`のwandbステップは3になります。

```python
for input, ground_truth in data:
    ...
    wandb.log("train/loss": 0.1)  
    wandb.log("eval/loss": 0.2)
```

これを回避するために、x軸ステップを明示的に定義することをお勧めします。`wandb.define_metric`でx軸を定義でき、これは`wandb.init`が呼び出された後一度だけ行う必要があります:

```
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

グロブパターン「\*」は、すべてのメトリクスがチャートのx軸に「global_step」を使用することを意味します。特定のメトリクスのみを「global_step」に対してログに記録したい場合は、それを指定できます。

```
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric`を呼び出したので、`wandb.log`を呼び出すたびに`step_metric`である「global_step」とメトリクスをログ化する必要があります:

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({"global_step": step, "train/loss": 0.1})
    wandb.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合（例えば「global_step」が検証ループ中に利用できない場合）、前回のログされた「global_step」の値は自動的にwandbによって使用されます。この場合、必要な時点で定義されるようにメトリクスの初期値をログ化するようにしてください。

#### 画像、テーブル、テキスト、音声などのログ

メトリクスに加えて、プロット、ヒストグラム、テーブル、テキスト、画像、動画、音声、3Dなどのメディアをログ化できます。

データをログ化する際の考慮事項には次のようなものがあります:

* メトリクスはどのくらいの頻度でログ化すべきか？オプションにするべきか？
* どの種類のデータが視覚化に役立つか？
  * 画像の場合、予測結果やセグメンテーションマスクなどをログ化し、時間経過による進化の様子を確認できます。
  * テキストの場合、後で探索するための予測結果のテーブルをログ化できます。

メディア、オブジェクト、プロットなどの完全なガイドについては、[wandb.logでデータをログ](https://docs.wandb.ai/guides/track/log)を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークの場合、以下のワークフローのいずれかを適応できます:

* "main"プロセスを検出し、`wandb`を唯一そこで使用する。他のプロセスから来る必要なデータは最初にmainプロセスにルーティングされる必要があります（このワークフローが推奨されます）。
* すべてのプロセスで`wandb`を呼び出し、すべてに同じ一意の`group`名を付けて自動グループ化する

詳細については、[ログ分散トレーニング実験](../track/log/distributed-training.md)を参照してください。

### モデルチェックポイントおよびその他のログ

フレームワークがモデルやデータセットを使用または生成する場合、wandbを使用してそれらをログし、W&B Artifactsを通じてパイプライン全体を自動的に監視できます。

![W&Bに保存されたデータセットとモデルチェックポイント](/images/integrations/integrations_add_any_lib_dag.png)

Artifactsを使用する際、ユーザーに次のことを定義させるのは有用ですが、必須ではありません:

* モデルチェックポイントやデータセットをログ化する能力（必要に応じてオプションにする場合）
* 入力として使用されるアーティファクトのパス/リファレンス。例えば「user/project/artifact」
* Artifactsのログ頻度

#### モデルチェックポイントのログ

モデルチェックポイントをW&Bにログ化できます。ユニークな`wandb` Run IDを利用して、出力モデルチェックポイントに名前を付け、Runs間で区別できるようにします。また、有用なメタデータを追加できます。さらに、以下のように各モデルにエイリアスを追加することも可能です:

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f“model-{wandb.run.id}”, 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # モデル重みが保存されるローカルディレクトリ

aliases = ["best", "epoch_10"] 
wandb.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアスの作成方法の詳細については、[カスタムエイリアスを作成](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias)を参照してください。

出力アーティファクトを任意の頻度（例えば、各エポックごと、500ステップごとなど）でログ化でき、アーティファクトは自動的にバージョン管理されます。

#### 学習済みモデルやデータセットのログと追跡

トレーニングに使用されるアーティファクト（学習済みモデルやデータセットなど）をログできます。以下のスニペットは、上記のグラフに示されるようにアーティファクトをログし、それを実行中のRunの入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
wandb.use_artifact(artifact_input_data)
```

#### W&B Artifactのダウンロード

アーティファクト（データセット、モデルなど）を再利用し、`wandb`がローカルにコピーをダウンロードします（キャッシュされます）:

```python
artifact = wandb.run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifactsは、W&BのArtifactsセクションにあり、自動生成されたエイリアス（「latest」、「v2」、「v3」）やログ時に手動で生成されたエイリアス（「best_accuracy」など）で参照できます。

分散環境や単純な推論のために（`wandb.init`を使用して）`wandb` runを作成せずにArtifactをダウンロードするには、[wandb API](https://docs.wandb.ai/ref/python/public-api)でアーティファクトを参照できます：

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Artifactsのダウンロードと使用](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact)を参照してください。

### ハイパーパラメータチューニング

ライブラリがW&Bハイパーパラメータチューニングを活用したい場合、[W&B Sweeps](https://docs.wandb.ai/guides/sweeps)をライブラリに追加できます。

### 高度なインテグレーション

以下のインテグレーションで高度なW&Bインテグレーションの例を見ることができます。ほとんどのインテグレーションはこれほど複雑ではありません：

