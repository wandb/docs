---
title: Add wandb to any library
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## あらゆるライブラリに wandb を追加する

このガイドでは、W&B を Python ライブラリに統合して、強力な 実験管理 、GPU およびシステム監視、モデルチェックポイントなどを独自のライブラリで利用するためのベストプラクティスを紹介します。

{{% alert %}}
W&B の使用方法をまだ学習中の場合は、このドキュメントの他の W&B ガイド（[実験管理]({{< relref path="/guides/models/track" lang="ja" >}})など）を参照してから、読み進めることをお勧めします。
{{% /alert %}}

以下では、作業しているコードベースが単一の Python トレーニングスクリプトまたは Jupyter ノートブックよりも複雑な場合のベストなヒントとベストプラクティスについて説明します。取り上げるトピックは次のとおりです。

* セットアップの要件
* ユーザーログイン
* wandb Run の開始
* Run の設定の定義
* W&B へのログ記録
* 分散トレーニング
* モデルのチェックポイントなど
* ハイパーパラメータの チューニング
* 高度なインテグレーション

### セットアップの要件

開始する前に、ライブラリの依存関係に W&B が必要かどうかを決定します。

#### インストール時に W&B を必須にする

W&B Python ライブラリ（`wandb`）を依存関係ファイルに追加します。たとえば、`requirements.txt` ファイルに追加します。

```python
torch==1.8.0
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK（`wandb`）をオプションにするには、次の 2 つの方法があります。

A. ユーザーが手動でインストールせずに `wandb` 機能を使用しようとしたときにエラーを発生させ、適切なエラーメッセージを表示します。

```python
try:
    import wandb
except ImportError:
    raise ImportError(
        "現在インストールされていない wandb を使用しようとしています。"
        "pip install wandb を使用してインストールしてください"
    )
```

B. Python パッケージを構築している場合は、`pyproject.toml` ファイルに `wandb` をオプションの依存関係として追加します。

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

#### APIキー を作成する

APIキー は、クライアントまたはマシンを W&B に対して認証します。APIキー は、ユーザープロファイルから生成できます。

{{% alert %}}
より合理的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にあるユーザープロファイルアイコンをクリックします。
2. **ユーザー設定**を選択し、**APIキー**セクションまでスクロールします。
3. **表示**をクリックします。表示された APIキー をコピーします。APIキー を非表示にするには、ページをリロードします。

#### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. APIキー に `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})を設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

2. `wandb` ライブラリをインストールしてログインします。

    ```shell
    pip install wandb

    wandb login
    ```

{{% /tab %}}

{{% tab header="Python" value="python" %}}

```bash
pip install wandb
```
```python
import wandb
wandb.login()
```

{{% /tab %}}

{{% tab header="Python ノートブック" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

ユーザーが上記の手順に従わずに初めて wandb を使用する場合、スクリプトが `wandb.init` を呼び出すと、自動的にログインするように求められます。

### Run を開始する

W&B Run は、W&B によってログに記録される計算の単位です。通常、トレーニング実験ごとに 1 つの W&B Run を関連付けます。

W&B を初期化し、コード内で Run を開始します。

```python
wandb.init()
```

必要に応じて、プロジェクトの名前を指定したり、コード内の `wandb_project` などのパラメータを使用してユーザー自身に設定させたり、エンティティパラメータに `wandb_entity` などのユーザー名または Team 名を設定させたりすることもできます。

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init` をどこに配置するか？

ライブラリはできるだけ早く W&B Run を作成する必要があります。エラーメッセージを含むコンソールへの出力はすべて W&B Run の一部としてログに記録されるためです。これにより、デバッグが容易になります。

#### `wandb` をオプションとしてライブラリを実行する

ユーザーがライブラリを使用するときに `wandb` をオプションにする場合は、次のいずれかを実行できます。

* 次のような `wandb` フラグを定義します。

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
trainer = my_trainer(..., use_wandb=True)
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
python train.py ... --use-wandb
```
{{% /tab %}}

{{< /tabpane >}}

* または、`wandb.init` で `wandb` を `disabled` に設定します。

{{< tabpane text=true >}}

{{% tab header="Python" value="python" %}}

```python
wandb.init(mode="disabled")
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
export WANDB_MODE=disabled
```

または

```bash
wandb disabled
```
{{% /tab %}}

{{< /tabpane >}}

* または、`wandb` をオフラインに設定します。これは、`wandb` は引き続き実行されますが、インターネット経由で W&B に通信を試行しないことに注意してください。

{{< tabpane text=true >}}

{{% tab header="環境変数" value="environment" %}}

```bash
export WANDB_MODE=offline
```

または

```python
os.environ['WANDB_MODE'] = 'offline'
```
{{% /tab %}}

{{% tab header="Bash" value="bash" %}}

```bash
wandb offline
```
{{% /tab %}}

{{< /tabpane >}}

### Run の config を定義する
`wandb` Run の config を使用すると、W&B Run を作成するときにモデル、データセットなどの メタデータ を提供できます。この情報を使用して、さまざまな実験を比較し、主な違いをすばやく理解できます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs テーブル" >}}

ログに記録できる一般的な config パラメータは次のとおりです。

* モデル名、バージョン、アーキテクチャパラメータなど。
* データセット名、バージョン、トレーニング/検証の例の数など。
* 学習率、バッチサイズ、 オプティマイザー などのトレーニングパラメータ。

次の コードスニペット は、config をログに記録する方法を示しています。

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run の config を更新する
config を更新するには、`wandb.config.update` を使用します。パラメータがディクショナリの定義後に取得された場合に、設定ディクショナリを更新すると便利です。たとえば、モデルのインスタンス化後にモデルのパラメータを追加する場合があります。

```python
wandb.config.update({"model_parameters": 3500})
```

config ファイルの定義方法の詳細については、[wandb.config で実験を設定する]({{< relref path="/guides/models/track/config" lang="ja" >}})を参照してください。

### W&B にログを記録する

#### メトリクス をログに記録する

キーの値が メトリクス の名前であるディクショナリを作成します。このディクショナリオブジェクトを[`wandb.log`]({{< relref path="/guides/models/track/log" lang="ja" >}})に渡します。

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data:
        prediction = model(input)
        loss = loss_fn(prediction, ground_truth)
        metrics = { "loss": loss }
        wandb.log(metrics)
```

多数の メトリクス がある場合は、`train/...` や `val/...` などの メトリクス 名にプレフィックスを使用すると、UI で自動的にグループ化できます。これにより、トレーニングと検証の メトリクス 、または分離したい他の メトリクス タイプについて、W&B Workspace に個別のセクションが作成されます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5,
    "val/accuracy": 0.7
}
wandb.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="2 つの個別のセクションがある W&B Workspace" >}}

`wandb.log` の詳細については、[wandb.log でデータをログに記録する]({{< relref path="/guides/models/track/log" lang="ja" >}})を参照してください。

#### x 軸のずれを防ぐ

同じトレーニングステップに対して `wandb.log` を複数回呼び出す必要がある場合があります。wandb SDK には、`wandb.log` が呼び出されるたびにインクリメントされる独自の内部ステップカウンターがあります。これは、wandb ログカウンターがトレーニングループのトレーニングステップと一致していない可能性があることを意味します。

これを回避するには、x 軸のステップを具体的に定義することをお勧めします。`wandb.define_metric` で x 軸を定義できます。これは、`wandb.init` が呼び出された後、1 回だけ実行する必要があります。

```python
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

グロブパターン `*` は、すべての メトリクス がチャートで `global_step` を x 軸として使用することを意味します。特定の メトリクス のみを `global_step` に対してログに記録する場合は、代わりにそれらを指定できます。

```python
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric` を呼び出したので、`wandb.log` を呼び出すたびに、 メトリクス と `step_metric`、`global_step` をログに記録するだけです。

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({"global_step": step, "train/loss": 0.1})
    wandb.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合（たとえば、検証ループ中に「global_step」を使用できない場合）、「global_step」に対して以前にログに記録された値が wandb によって自動的に使用されます。この場合、 メトリクス の初期値をログに記録して、必要なときに定義されるようにしてください。

#### 画像、 テーブル 、音声などをログに記録する

メトリクス に加えて、プロット、ヒストグラム、 テーブル 、テキスト、および画像、ビデオ、音声、3D などのメディアをログに記録できます。

データをログに記録する際の考慮事項には、次のものがあります。

* メトリクス をログに記録する頻度はどれくらいですか？オプションにする必要がありますか？
* 視覚化に役立つ可能性のあるデータの種類は何ですか？
  * 画像の場合、時間の経過に伴う変化を確認するために、サンプル予測、セグメンテーションマスクなどをログに記録できます。
  * テキストの場合、後で探索するために、サンプル予測の テーブル をログに記録できます。

メディア、オブジェクト、プロットなどのログ記録に関する完全なガイドについては、[wandb.log でデータをログに記録する]({{< relref path="/guides/models/track/log" lang="ja" >}})を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークの場合、次のいずれかの ワークフロー を適用できます。

* どのプロセスが「メイン」プロセスであるかを検出し、そこで `wandb` のみを使用します。他のプロセスから送信される必要なデータは、最初にメインプロセスにルーティングする必要があります。（この ワークフロー をお勧めします）。
* すべてのプロセスで `wandb` を呼び出し、すべてに同じ一意の `group` 名を付けて自動的にグループ化します。

詳細については、[分散トレーニング実験をログに記録する]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})を参照してください。

### モデル のチェックポイントなどをログに記録する

フレームワークが モデル またはデータセットを使用または生成する場合、それらを完全に追跡できるようにログに記録し、W&B Artifacts を介してパイプライン全体を wandb で自動的に監視できます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="W&B に保存されたデータセットとモデルチェックポイント" >}}

Artifacts を使用する場合、ユーザーが定義できるようにすると役立つ場合がありますが、必須ではありません。

* モデル のチェックポイントまたはデータセットをログに記録する機能（オプションにする場合）。
* 入力として使用される Artifacts のパス/参照（存在する場合）。たとえば、`user/project/artifact` などです。
* Artifacts をログに記録する頻度。

#### モデル のチェックポイントをログに記録する

モデル のチェックポイントを W&B にログに記録できます。一意の `wandb` Run ID を利用して、出力 モデル のチェックポイントに名前を付けて、Run 間で区別すると便利です。また、役立つ メタデータ を追加することもできます。さらに、以下に示すように、各 モデル に エイリアス を追加することもできます。

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800}

artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                metadata=metadata,
                type="model"
                )
artifact.add_dir("output_model") # モデル の重みが保存されているローカルディレクトリー

aliases = ["best", "epoch_10"]
wandb.log_artifact(artifact, aliases=aliases)
```

カスタム エイリアス の作成方法については、[カスタム エイリアス を作成する]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}})を参照してください。

出力 Artifacts は任意の頻度（たとえば、エポックごと、500 ステップごとなど）でログに記録でき、自動的にバージョン管理されます。

#### 学習済み モデル またはデータセットをログに記録して追跡する

トレーニングへの入力として使用される Artifacts （学習済み モデル やデータセットなど）をログに記録できます。次の スニペット は、Artifacts をログに記録し、上記のグラフに示すように、それを実行中の Run への入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
wandb.use_artifact(artifact_input_data)
```

#### Artifacts をダウンロードする

Artifacts （データセット、 モデル など）を再利用すると、`wandb` はローカルにコピーをダウンロードします（そしてキャッシュします）。

```python
artifact = wandb.run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションにあり、自動的に生成される エイリアス （`latest`、`v2`、`v3`）またはログ記録時に手動で生成される エイリアス （`best_accuracy` など）で参照できます。

分散環境または単純な推論など、`wandb` Run（`wandb.init` を使用）を作成せずに Artifacts をダウンロードするには、代わりに[wandb API]({{< relref path="/ref/python/public-api" lang="ja" >}})で Artifacts を参照できます。

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Artifacts をダウンロードして使用する]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

### ハイパーパラメータを チューニング する

ライブラリで W&B ハイパーパラメータ チューニング 、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})を活用したい場合は、W&B Sweeps をライブラリに追加することもできます。

### 高度なインテグレーション

高度な W&B インテグレーション がどのように見えるかを、次のインテグレーション で確認することもできます。ほとんどのインテグレーション はこれほど複雑ではありません。

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)
