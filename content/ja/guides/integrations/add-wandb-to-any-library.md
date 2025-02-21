---
title: Add wandb to any library
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を追加する

このガイドでは、強力な 実験管理、GPU およびシステム監視、モデルチェックポイントなどを自分のライブラリで取得するために、Python ライブラリに W&B を統合するためのベストプラクティスを紹介します。

{{% alert %}}
まだ W&B の使い方を学んでいる場合は、このドキュメント内の他の W&B ガイド、例えば [実験管理]({{< relref path="/guides/models/track" lang="ja" >}})などを先に読むことをお勧めします。
{{% /alert %}}

ここでは、1つの Python トレーニングスクリプトまたは Jupyter ノートブックよりも複雑なコードベースで作業する際のベストなヒントとベストプラクティスを紹介します。取り上げるトピックは以下のとおりです：

* セットアップ要件
* ユーザーログイン
* wandb Run を開始する
* Run Config を定義する
* W&B へのログ
* 分散トレーニング
* モデルチェックポイントとその他
* ハイパーパラメータチューニング
* 高度なインテグレーション

### セットアップ要件

始める前に、ライブラリの依存関係に W&B を要求するかどうかを決定します：

#### インストール時に W&B を要求する

W&B Pythonライブラリ (`wandb`) を依存関係ファイルに追加します。例として `requirements.txt` ファイルに以下を追加します：

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK (`wandb`) をオプションにするには、2つの方法があります：

A. インストールせずに `wandb` 機能を使用しようとした際にエラーを発生させ、適切なエラーメッセージを表示します：

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "wandb を使用しようとしていますが、現在インストールされていません。"
        "pip install wandbを使用してインストールしてください。"
    ) 
```

B. `pyproject.toml` ファイルで、`wandb` をオプションの依存関係として追加します（Python パッケージを構築している場合）：

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

#### APIキーを作成する

APIキーは、クライアントまたはマシンを W&B に認証します。APIキーはユーザープロフィールから生成できます。

{{% alert %}}
よりスムーズなアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキーを生成することができます。表示された APIキーをコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
2. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
3. **Reveal** をクリックします。表示された APIキーをコピーし、APIキーを非表示にするにはページを再読み込みします。

#### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには：

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. APIキーを [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_API_KEY` に設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールしてログインします。

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

wandb を初めて使用するユーザーは、スクリプトが `wandb.init` を呼び出すと自動的にログインを促されます。

### Run を開始する

W&B Run は、W&Bによってログに記録される計算の単位です。通常、トレーニング実験ごとに1つの W&B Run を関連付けます。

以下で W&B を初期化し、コード内で Run を開始します：

```python
wandb.init()
```

プロジェクトに名前を付けるか、ユーザーにコード内のパラメータ `wandb_project` と、エンティティパラメータのためのユーザー名またはチーム名 `wandb_entity` を設定させることができます：

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init` の配置場所

ライブラリは W&B Run を可能な限り早く作成する必要があります。コンソールの出力（エラーメッセージを含む）はすべて W&B Run の一部としてログに記録されるため、デバッグが簡単になります。

#### ライブラリを `wandb` をオプションとして実行する

ユーザーがライブラリを使用する際に `wandb` をオプションにしたい場合、以下のいずれかを選択できます：

* `wandb` フラグを定義します：

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

* または、`wandb.init` で `wandb` を `disabled` に設定します：

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

* または、`wandb` をオフラインに設定します。これは `wandb` をまだ実行することを意味し、W&B へのインターネット通信を試みないだけです：

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

### Run Config を定義する
`wandb` run config を使って、モデル、データセットなどに関するメタデータを W&B Run 作成時に提供できます。この情報を使用して異なる実験を比較し、主要な違いを素早く理解できます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

記録できる典型的なコンフィグパラメータは以下の通りです：

* モデル名、バージョン、アーキテクチャーパラメータなど
* データセット名、バージョン、訓練/検証例の数など
* トレーニングパラメータ（学習率、バッチサイズ、オプティマイザーなど）

以下のコードスニペットはコンフィグを記録する方法を示しています：

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run Config を更新する
`wandb.config.update` を使用してコンフィグを更新します。辞書が定義された後に取得されたパラメータには設定を更新することが有用です。例えば、モデルがインスタンス化された後にモデルのパラメータを追加する場合などです。

```python
wandb.config.update({"model_parameters": 3500})
```

コンフィグファイルの定義方法については、[Configure Experiments with wandb.config]({{< relref path="/guides/models/track/config" lang="ja" >}}) を参照してください。

### W&B へのログ

#### メトリクスのログ

キー値がメトリクスの名前である辞書を作成します。この辞書オブジェクトを [`wandb.log`]({{< relref path="/guides/models/track/log" lang="ja" >}}) に渡します：

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        wandb.log(metrics)
```

大量のメトリクスがある場合、UIで自動的にグループ化されるよう、メトリクス名に接頭辞を使用することをお勧めします。たとえば `train/...` や `val/...` などです。これにより、W&B ワークスペースにトレーニングと検証のメトリクスに対する別々のセクションが作成されます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
wandb.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="A W&B Workspace with 2 separate sections" >}}

`wandb.log` の詳細については、[Log Data with wandb.log]({{< relref path="/guides/models/track/log" lang="ja" >}}) を参照してください。

#### x軸の不整合を防ぐ

同じトレーニングステップで `wandb.log` を複数回呼び出す必要がある場合があります。wandb SDKには独自の内部ステップカウンターがあり、`wandb.log` の呼び出しごとにインクリメントされます。これにより、wandb のログカウンターがトレーニングループ内のトレーニングステップと一致しない可能性があります。

これを回避するために、x軸のステップを明示的に定義することをお勧めします。x軸は `wandb.define_metric` で定義できます。これは `wandb.init` が呼び出された後に1度だけ実行すればよいです：

```python
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

グロブパターン `*` は全メトリクスに対して `global_step` をチャートの x軸に利用することを意味します。特定のメトリクスのみを `global_step` に対してログしたい場合は、それらを指定できます：

```python
wandb.define_metric("train/loss", step_metric="global_step")
```

現在、`wandb.define_metric` を呼び出した後、メトリクスと `step_metric`、`global_step`を `wandb.log` の各呼び出し時にログに記録するだけです：

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({"global_step": step, "train/loss": 0.1})
    wandb.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合、例えば "global_step" が検証ループ中に利用できない場合、wandb によって先にログされた "global_step" の値が自動的に使用されます。この場合、必要なときにメトリクスが定義されているように初期値をログに記録することを確認してください。

#### 画像、テーブル、オーディオなどをログする

メトリクスに加え、プロット、ヒストグラム、テーブル、テキスト、メディアなど（画像、動画、オーディオ、3D など）を記録できます。

データをログに記録する際の考慮点には以下のものがあります：

* メトリクスはどれくらいの頻度で記録するべきか？オプションにするべきか？
* 視覚化するのに役立つデータのタイプは何か？
  * 画像の場合、サンプル予測、セグメンテーションマスクなどを記録して、時間の経過とともに進化を見ることができます。
  * テキストの場合、後で探索するためのサンプル予測のテーブルを記録できます。

[Log Data with wandb.log]({{< relref path="/guides/models/track/log" lang="ja" >}}) でメディア、オブジェクト、プロット、その他のログについての完全ガイドを参照してください。

### 分散トレーニング

分散環境でのフレームワークをサポートするために、以下のワークフローのいずれかを適応させられます：

* どのプロセスが "メイン" であるかを検出し、`wandb` をそこだけで使用します。他のプロセスから必要なデータはまずメインプロセスにルーティングされる必要があります。（このワークフローを推奨します）。
* すべてのプロセスで `wandb` を呼び出し、全てに同じ一意の `group` 名を与えて自動的にグループ化します。

詳細については [Log Distributed Training Experiments]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。

### モデルのチェックポイントとその他をログに記録する

フレームワークがモデルまたはデータセットを使用または生成する場合、wandb を使用してそれらを完全に追跡可能にログに記録し、W&B Artifacts を通してパイプライン全体を自動的にモニターさせることができます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="Stored Datasets and Model Checkpoints in W&B" >}}

Artifacts を使用する場合、ユーザーに設定を定義させると便利ですが、必須ではありません：

* モデルチェックポイントやデータセットをログに記録する機能（オプションにしたい場合）
* 入力として使用されるアーティファクトのパス/参照。たとえば、`user/project/artifact` です。
* Artifacts のログ記録の頻度。

#### モデルチェックポイントをログに記録する

モデルチェックポイントを W&B に記録できます。固有の `wandb` Run ID を活用して出力モデルチェックポイントを名前付けすることで、Run 間でそれらを識別することができます。また、有用なメタデータを追加することも可能です。さらに、以下に示すように各モデルにエイリアスを追加することもできます：

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # モデルのウェイトが保存されているローカルディレクトリー

aliases = ["best", "epoch_10"] 
wandb.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアスを作成する方法については、[Create a Custom Alias]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}}) を参照してください。

Artifacts の出力を任意の頻度でログに記録できます（たとえば、各エポック、500ステップごとなど）であり、自動的にバージョン管理されます。

#### 学習済みモデルやデータセットをログおよび追跡する

訓練の入力として使用されるアーティファクト（学習済みモデルやデータセットなど）をログに記録できます。以下のスニペットは、アーティファクトをログに記録し、上の図のように進行中の Run に入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
wandb.use_artifact(artifact_input_data)
```

#### アーティファクトをダウンロードする

アーティファクト（データセット、モデルなど）を再利用し、wandb がローカルにコピーをダウンロード（およびキャッシュ）します：

```python
artifact = wandb.run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションにあり、自動的に生成されたエイリアス（`latest`、`v2`、`v3`）や、ログ時に手動で指定されたエイリアス（`best_accuracy` など）で参照できます。

アーティファクトを `wandb` run を作成せずにダウンロードするには（`wandb.init` なし）、たとえば分散環境や単純な推論のために、代わりに [wandb API]({{< relref path="/ref/python/public-api" lang="ja" >}}) を使用してアーティファクトを参照することができます：

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Download and Use Artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}}) を参照してください。

### ハイパーパラメータをチューニングする

ライブラリが W&B ハイパーパラメータチューニングを利用したい場合、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) をライブラリに追加することもできます。

### 詳細なインテグレーション

以下のインテグレーションで、詳細な W&B インテグレーションがどのように見えるかも確認できます。ほとんどのインテグレーションはこれほど複雑ではないことに注意してください：

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)
```