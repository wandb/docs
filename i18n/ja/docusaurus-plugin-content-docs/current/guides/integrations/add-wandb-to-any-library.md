import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 任意のライブラリにwandbを追加

このガイドでは、PythonライブラリにWeights & Biasesを統合する方法と、自分のライブラリに強力な実験トラッキング、GPUおよびシステム監視、モデルチェックポイントなどを取得するためのベストプラクティスを提供します。

:::note
まだW&Bの使い方を学んでいる場合は、これらのドキュメント内の他のW&Bガイドを探して、[実験トラッキング](https://docs.wandb.ai/guides/track)などを読むことをお勧めします。
:::

以下では、作業中のコードベースが単一のPythonトレーニングスクリプトやJupyterノートブックより複雑な場合のベストティップスとベストプラクティスを紹介します。カバーされるトピックは以下の通りです。

* セットアップ要件
* ユーザーログイン
* wandbランの開始
* ランの設定の定義
* Weights & Biasesへのログ記録
* 分散トレーニング
* モデルチェックポイントなど
* ハイパーパラメータチューニング
* 高度な統合
### セットアップ要件

始める前に、ライブラリの依存関係にW&Bを必要とするかどうかを決定してください。

#### インストール時にW&Bを必要とする

W&B Pythonライブラリ（`wandb`）を依存関係ファイルに追加します。例えば、`requirements.txt`ファイルに追加します。

```python
torch==1.8.0 
...
wandb==0.13.*
```
#### W&Bのインストールをオプションにする方法

W&B SDK（`wandb`）をオプションにする方法は2つあります。

A. ユーザーが`wandb`を手動でインストールせずに機能を使用しようとするとエラーを発生させ、適切なエラーメッセージを表示します。

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "wandbを使用しようとしていますが、現在インストールされていません"
        "pip install wandbを使用してインストールしてください"
    ) 
```
B. Pythonパッケージをビルドしている場合、`pyproject.toml`ファイルに`wandb`をオプションの依存関係として追加します。

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

ユーザーがW＆Bにログインする方法はいくつかあります。

<Tabs
  defaultValue="bash"
  values={[
    {label: 'Bash', value: 'bash'},
    {label: 'Notebook', value: 'notebook'},
    {label: 'Environment Variable', value: 'environment'},
  ]}>
  <TabItem value="bash">
ターミナルでbashコマンドを使ってW&Bにログインします

```bash
wandb login $MY_WANDB_KEY
```
  </TabItem>
  <TabItem value="notebook">
JupyterやColabノートブック内であれば、以下のようにしてW＆Bにログインします
```python
import wandb
wandb.login
```
  </TabItem>
  <TabItem value="環境">

APIキーの[W&B環境変数](../track/environment-variables.md)を設定します

```bash
export WANDB_API_KEY=$YOUR_API_KEY
```

または
```
os.environ['WANDB_API_KEY'] = "abc123..."
```
  </TabItem>
</Tabs>


上記の手順を何も踏まずにユーザーがwandbを初めて使う場合、スクリプトが`wandb.init`を呼び出すと自動的にログインが求められます。

### wandb Runの開始

W&B Runは、Weights & Biasesによって記録される計算の単位です。通常、1つのW&B Runを1つのトレーニング実験に関連付けます。

コード内でW&Bを初期化し、Runを開始するには：
以下は翻訳するMarkdownテキストのチャンクです。追加のコメントなどせずに、翻訳されたテキストのみを返してください。テキスト：

```python
wandb.init()
```

オプションとして、プロジェクトに名前を付けることができますし、ユーザーがパラメータ（`wandb_project` など）をコードに指定することで自分で設定することもできます。同様に、ユーザ名やチーム名を `wandb_entity` として指定できます。

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init` をどこに配置するか？

ライブラリはできるだけ早い段階でW&B Runを作成するべきです。なぜなら、コンソールの出力（エラーメッセージを含む）がW&B Runの一部として記録されるため、デバッグが容易になるからです。
#### `wandb`をオプションとしてライブラリをrunする

ユーザーがあなたのライブラリを使うときに`wandb`をオプションにしたい場合、以下のいずれかを行うことができます:

* `wandb` フラグを定義する方法:

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

* または、`wandb.init`で`wandb`を無効に設定します。

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
* または、`wandb`をオフラインに設定してください - これでも`wandb`は動作しますが、インターネット経由でWeights & Biasesに通信しようとはしません

<Tabs
  defaultValue="environment"
  values={[
    {label: '環境変数', value: 'environment'},
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

### W&B Runの設定を定義する

`wandb` runの設定を使用して、W&B Runを作成する際に、モデルやデータセットなどに関するメタデータを提供できます。この情報を利用して、異なる実験を比較し、主な違いをすばやく把握できます。
![Weights & Biases Runs table](/images/integrations/integrations_add_any_lib_runs_page.png)

ログできる一般的な設定パラメータには以下のようなものがあります：

* モデル名、バージョン、アーキテクチャーのパラメータなど
* データセット名、バージョン、トレーニング/バリデーションの例の数など
* ラーニングレート、バッチサイズ、オプティマイザーなどのトレーニングパラメータ

以下のコードスニペットは、設定をログに記録する方法を示しています：

```python
config = {"batch_size":32, …}
wandb.init(…, config=config)
```
#### wandb設定の更新

`wandb.config.update`を使って設定を更新します。設定ディクショナリが定義された後にパラメータが得られた場合、たとえばモデルがインスタンス化された後にモデルのパラメータを追加する場合など、設定ディクショナリを更新すると便利です。

```python
wandb.config.update({"model_parameters": 3500})
```

設定ファイルの定義方法の詳細については、[wandb.configを使った実験の設定](https://docs.wandb.ai/guides/track/config) を参照してください。

### Weights & Biasesへのログ記録

#### メトリクスのログ

キー値がメトリック名のディクショナリを作成します。このディクショナリオブジェクトを[`wandb.log`](https://docs.wandb.ai/guides/track/log)に渡します：
```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        wandb.log(metrics)
```

たくさんのメトリクスがある場合は、メトリクス名にプレフィックス（`train/...` や `val/...`など）を使用して、UIで自動的にグループ化できます。これにより、W&Bのワークスペースにトレーニングと検証のメトリクス、または他の区別したいメトリクスタイプごとに個別のセクションが作成されます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
wandb.log(metrics)
```
![Weights & Biasesワークスペースには2つの別々のセクションがあります](/images/integrations/integrations_add_any_lib_log.png)

`wandb.log`についての詳細は、[wandb.logを使ってデータをログ](https://docs.wandb.ai/guides/track/log)を確認してください。

#### x軸のずれを防止

時には、同じトレーニングステップで複数回`wandb.log`を呼び出す必要があることがあります。wandb SDKには、`wandb.log`呼び出しの度にインクリメントされる内部ステップカウンタがあります。これは、wandbログカウンタがトレーニングループ内のトレーニングステップと整列していない可能性があることを意味します。

以下の例の初回では、`train/loss`の内部`wandb`ステップは0になりますが、`eval/loss`の内部`wandb`ステップは1になります。次の回では、`train/loss`は2になりますが、`eval/loss`のwandbステップは3になります。

```python
for input, ground_truth in data:
    ...
    wandb.log(“train/loss”: 0.1)  
    wandb.log(“eval/loss”: 0.2)
```
これを回避するために、x軸のステップを明示的に定義することをお勧めします。`wandb.define_metric`でx軸を定義できます。これは、`wandb.init`が呼び出された後に一度だけ行う必要があります。

```
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

グロブパターンの "\*" は、すべてのメトリクスがチャートのx軸に "global_step" を使用することを意味します。 "global_step"に対して特定のメトリクスのみをログに記録したい場合は、それらを指定できます。

```
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric`を呼び出した後は、`wandb.log`を呼び出すたびに、メトリクスと`step_metric`である "global_step" をログに記録するだけです。
```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({"global_step": step, "train/loss": 0.1})
    wandb.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合、例えば、検証ループ中に "global_step"が利用できない場合、wandbによって前回ログされた "global_step" の値が自動的に使用されます。 この場合、必要な時に定義されていることを確認するため、メトリクスの初期値をログしてください。

#### 画像、テーブル、テキスト、オーディオなどをログする

メトリクスに加えて、プロット、ヒストグラム、テーブル、テキスト、画像、ビデオ、オーディオ、3Dなどのメディアをログすることができます。

データをログする際のいくつかの注意点は次のとおりです。
* メトリックはどのくらいの頻度でログに記録すべきですか？オプションにするべきですか？
* どのようなデータが可視化に役立ちますか？
  * 画像の場合、サンプル予測やセグメンテーションマスクなどをログに記録して、時間の経過とともに進化を見ることができます。
  * テキストの場合、後で探索するためのサンプル予測の表をログに記録できます。

メディア、オブジェクト、プロットなどのログ記録に関する完全なガイドは [wandb.logを使ってデータをログに記録する](https://docs.wandb.ai/guides/track/log)を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークでは、以下のワークフローのいずれかに適応できます。

* 「メイン」プロセスがどれかを検出し、そこでのみ`wandb`を使用します。他のプロセスから必要なデータは、まずメインプロセスにルーティングされる必要があります（このワークフローが推奨されています）。
* すべてのプロセスで`wandb`を呼び出し、すべてに同じユニークな`group`名を付けることで自動的にグループ化します

詳細については [分散トレーニング実験のログ記録](../track/log/distributed-training.md) を参照してください。
### モデルチェックポイントとその他のロギング

フレームワークがモデルやデータセットを使用または生成する場合、W&B Artifactsを使用して完全なトレーサビリティを持たせ、wandbがW&B Artifactsを利用して開発フロー全体を自動監視できるようにすることができます。

![W&Bに保存されたデータセットとモデルチェックポイント](/images/integrations/integrations_add_any_lib_dag.png)

Artifactsを使用する際、以下の機能をユーザーに定義させることが役立ちますが、必須ではありません。

* モデルチェックポイントやデータセットをログに記録する機能（オプションで提供したい場合）
* 入力として使用されるアーティファクトのパス/リファレンス（存在する場合）。例えば、「user/project/artifact」
* アーティファクトのログ記録頻度

#### モデルチェックポイントのログ記録

W&Bにモデルチェックポイントをログ記録できます。ユニークな`wandb` Run IDを使用して出力モデルチェックポイントに名前を付け、Runごとに区別することが有益です。また、有用なメタデータを追加することもできます。さらに、以下に示すように、各モデルにエイリアスを追加することもできます。
```python
metadata = {“eval/accuracy”: 0.8, “train/steps”: 800} 

artifact = wandb.Artifact(
                name=f”model-{wandb.run.id}”, 
                metadata=metadata, 
                type=”model”
                ) 
artifact.add_dir(“output_model”) #モデルの重みが保存されているローカルディレクトリー

aliases = [“best”, “epoch_10”] 
wandb.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアスの作成方法については、[カスタムエイリアスの作成](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias)を参照してください。
出力されたアーティファクトは、任意の頻度（例えば、エポックごとや500ステップごとなど）でログに記録することができ、自動的にバージョン管理されます。

#### 事前学習済みモデルやデータセットのログとトラッキング

事前学習済みのモデルやデータセットなど、トレーニングへの入力に使われるアーティファクトをログに記録することができます。以下のスニペットは、アーティファクトをログに記録し、上記のグラフに示すように実行中のRunの入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name=”flowers”, type=”dataset”)
artifact_input_data.add_file(“flowers.npy”)
wandb.use_artifact(artifact_input_data)
```

#### W&Bアーティファクトのダウンロード

アーティファクト（データセット、モデルなど）を再利用し、`wandb`がローカルにコピー（およびキャッシュ）をダウンロードします：
```python
artifact = wandb.run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

アーティファクトは、W&Bのアーティファクトセクションで見つけることができ、自動的に生成されたエイリアス（“latest”, “v2”, “v3”）やログ記録時に手動で設定されたエイリアス（“best_accuracy”...）で参照することができます。

`wandb` runを作成せずにアーティファクトをダウンロードするには（例：`wandb.init`を介さない分散環境や単純な推論の場合）、代わりに[wandb API](https://docs.wandb.ai/ref/python/public-api) でアーティファクトを参照できます。

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Download and Use Artifacts](https://docs.wandb.ai/guides/artifacts/download-and-use-an-artifact)を参照してください。
### ハイパーパラメータチューニング

あなたのライブラリがW&Bのハイパーパラメータチューニングを活用したい場合、[W&Bスイープ](https://docs.wandb.ai/guides/sweeps)もライブラリに追加することができます。

### 上級インテグレーション

また、以下のインテグレーションで高度なW&Bインテグレーションの例を見ることができます。 ほとんどのインテグレーションはこれらほど複雑ではありませんのでご注意ください:

* [Hugging Faceトランスフォーマー `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)