---
title: Add wandb to any library
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を追加する

このガイドでは、強力な 実験管理 、GPU とシステム監視、モデルチェックポイントなど、独自のライブラリのための機能を W&B と統合するためのベストプラクティスを提供します。

{{% alert %}}
W&B の使用方法をまだ学習中の場合は、先に進む前に、[実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) など、これらのドキュメントにある他の W&B ガイドを確認することをお勧めします。
{{% /alert %}}

以下では、作業中のコードベースが単一の Python トレーニングスクリプトまたは Jupyter ノートブックよりも複雑な場合の、ベストなヒントとベストプラクティスについて説明します。取り上げるトピックは次のとおりです。

* セットアップ要件
* ユーザーログイン
* wandb の Run の開始
* Run の設定の定義
* W&B へのログ記録
* 分散トレーニング
* モデルチェックポイントなど
* ハイパーパラメータの チューニング
* 高度な インテグレーション

### セットアップ要件

開始する前に、ライブラリの依存関係に W&B を必須にするかどうかを決定します。

#### インストール時に W&B を必須とする

W&B Python ライブラリ（`wandb`）を依存関係ファイルに追加します。たとえば、`requirements.txt` ファイルに追加します。

```python
torch==1.8.0
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK（`wandb`）をオプションにするには、2つの方法があります。

A. ユーザーが手動でインストールせずに `wandb` 機能を使用しようとしたときにエラーを発生させ、適切なエラーメッセージを表示します。

```python
try:
    import wandb
except ImportError:
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    )
```

B. Python パッケージを構築している場合は、`wandb` をオプションの依存関係として `pyproject.toml` ファイルに追加します。

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

APIキー は、クライアントまたはマシンを W&B に対して認証します。 APIキー は、ユーザープロフィールから生成できます。

{{% alert %}}
より合理的なアプローチとして、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして APIキー を生成できます。表示された APIキー をコピーし、パスワードマネージャーなどの安全な場所に保存します。
{{% /alert %}}

1. 右上隅にあるユーザープロフィールアイコンをクリックします。
2. [**User Settings（ユーザー設定）**]を選択し、[**API Keys（APIキー）**]セクションまでスクロールします。
3. [**Reveal（表示）**]をクリックします。表示された APIキー をコピーします。 APIキー を非表示にするには、ページをリロードします。

#### `wandb` ライブラリをインストールしてログインする

`wandb` ライブラリをローカルにインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) を APIキー に設定します。

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

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

ユーザーが上記の手順に従わずに初めて wandb を使用する場合、スクリプトが `wandb.init` を呼び出すと、自動的にログインを求められます。

### Run を開始する

W&B Run は、W&B によって記録される計算の単位です。通常、トレーニング実験ごとに単一の W&B Run を関連付けます。

W&B を初期化し、コード内で Run を開始するには:

```python
run = wandb.init()
```

オプションで、プロジェクトの名前を指定したり、エンティティパラメーターのユーザー名やチーム名（`wandb_entity`）とともに、コード内の `wandb_project` などのパラメーターを使用して、ユーザー自身に設定させたりできます。

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run を終了するには、`run.finish()` を呼び出す必要があります。これがインテグレーションの設計で機能する場合は、Run をコンテキストマネージャーとして使用します。

```python
# When this block exits, it calls run.finish() automatically.
# If it exits due to an exception, it uses run.finish(exit_code=1) which
# marks the run as failed.
with wandb.init() as run:
    ...
```

#### `wandb.init` をいつ呼び出すか?

ライブラリは、W&B Run をできるだけ早く作成する必要があります。これは、エラーメッセージを含むコンソール内のすべての出力が W&B Run の一部として記録されるためです。これにより、デバッグが容易になります。

#### `wandb` をオプションの依存関係として使用する

ユーザーがライブラリを使用する際に `wandb` をオプションにしたい場合は、次のいずれかの方法があります。

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

* または、`wandb` をオフラインに設定します。これは、`wandb` を実行しますが、インターネット経由で W&B に通信しようとはしません。

{{< tabpane text=true >}}

{{% tab header="Environment Variable" value="environment" %}}

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

### Run の設定を定義する
`wandb` の Run の設定を使用すると、W&B Run の作成時にモデル、データセットなどに関するメタデータを提供できます。この情報を使用して、さまざまな実験を比較し、主な違いをすばやく理解できます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

ログに記録できる一般的な設定パラメーターは次のとおりです。

* モデル名、バージョン、アーキテクチャー パラメーターなど。
* データセット名、バージョン、トレーニング/検証の例の数など。
* 学習率、 バッチサイズ 、 オプティマイザー などのトレーニングパラメーター。

次のコードスニペットは、設定をログに記録する方法を示しています。

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run の設定を更新する
設定を更新するには、`run.config.update` を使用します。パラメーターが辞書の定義後に取得される場合に、設定辞書を更新すると便利です。たとえば、モデルのインスタンス化後にモデルのパラメーターを追加する場合があります。

```python
run.config.update({"model_parameters": 3500})
```

設定ファイルの定義方法の詳細については、[実験の設定]({{< relref path="/guides/models/track/config" lang="ja" >}}) を参照してください。

### W&B にログを記録する

#### メトリクス を記録する

キーの値が メトリクス の名前である辞書を作成します。この辞書オブジェクトを[`run.log`]({{< relref path="/guides/models/track/log" lang="ja" >}})に渡します。

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data:
        prediction = model(input)
        loss = loss_fn(prediction, ground_truth)
        metrics = { "loss": loss }
        run.log(metrics)
```

メトリクス がたくさんある場合は、`train/...` や `val/...` などの メトリクス 名にプレフィックスを使用することで、UI で自動的にグループ化できます。これにより、トレーニングと検証の メトリクス 、または分離したいその他の メトリクス タイプ用に、W&B Workspace に個別のセクションが作成されます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5,
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="A W&B Workspace with 2 separate sections" >}}

[`run.log` の詳細を見る]({{< relref path="/guides/models/track/log" lang="ja" >}})。

#### X軸のずれを防ぐ

同じトレーニングステップに対して `run.log` を複数回呼び出すと、wandb SDK は `run.log` を呼び出すたびに内部ステップカウンターをインクリメントします。このカウンターは、トレーニングループのトレーニングステップと一致しない場合があります。

この状況を回避するには、`run.define_metric` で X軸ステップを明示的に定義します。`wandb.init` を呼び出した直後に1回定義します。

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

グロブパターン `*` は、すべての メトリクス がチャートで `global_step` を X軸として使用することを意味します。特定の メトリクス のみを `global_step` に対してログに記録する場合は、代わりにそれらを指定できます。

```python
run.define_metric("train/loss", step_metric="global_step")
```

次に、`run.log` を呼び出すたびに、 メトリクス 、`step` メトリクス 、および `global_step` をログに記録します。

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

たとえば、検証ループ中に「global_step」が利用できないなど、独立したステップ変数にアクセスできない場合、「global_step」の以前にログに記録された値が wandb によって自動的に使用されます。この場合、メトリクス に必要なときに定義されるように、 メトリクス の初期値をログに記録してください。

#### 画像、テーブル、音声などをログに記録する

メトリクス に加えて、プロット、ヒストグラム、テーブル、テキスト、および画像、ビデオ、オーディオ、3D などのメディアをログに記録できます。

データをログに記録する際の考慮事項は次のとおりです。

* メトリクス をログに記録する頻度はどのくらいですか? オプションにする必要がありますか?
* 視覚化に役立つデータの種類は何ですか?
  * 画像の場合は、サンプル予測、セグメンテーションマスクなどをログに記録して、時間の経過に伴う変化を確認できます。
  * テキストの場合は、サンプル予測のテーブルをログに記録して、後で調べることができます。

メディア、オブジェクト、プロットなどの[ログ記録の詳細]({{< relref path="/guides/models/track/log" lang="ja" >}})をご覧ください。

### 分散トレーニング

分散環境をサポートするフレームワークの場合は、次のいずれかの ワークフロー を採用できます。

* どの プロセス が「メイン」 プロセス であるかを検出し、そこで `wandb` のみを使用します。他の プロセス からの必要なデータは、最初にメイン プロセス にルーティングする必要があります（この ワークフロー を推奨します）。
* すべての プロセス で `wandb` を呼び出し、すべてに同じ一意の `group` 名を付けて自動的にグループ化します。

詳細については、[分散トレーニング実験のログを記録する]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})を参照してください。

### モデルチェックポイントなどを記録する

フレームワークがモデルまたはデータセットを使用または生成する場合は、それらをログに記録して完全なトレーサビリティを実現し、W&B Artifacts を介して パイプライン 全体を wandb で自動的に監視できます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="Stored Datasets and Model Checkpoints in W&B" >}}

Artifacts を使用する場合、ユーザーに次のことを定義させることは役立つかもしれませんが、必須ではありません。

* モデルチェックポイントまたはデータセットをログに記録する機能（オプションにする場合）。
* 入力として使用される Artifact のパス/参照（ある場合）。たとえば、`user/project/artifact` です。
* Artifacts をログに記録する頻度。

#### モデルチェックポイント を記録する

モデルチェックポイント を W&B にログを記録できます。一意の `wandb` Run ID を利用して出力 モデルチェックポイント に名前を付け、Run 間で区別すると便利です。また、有用な メタデータ を追加することもできます。さらに、以下に示すように、各モデルに エイリアス を追加することもできます。

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800}

artifact = wandb.Artifact(
                name=f"model-{run.id}",
                metadata=metadata,
                type="model"
                )
artifact.add_dir("output_model") # local directory where the model weights are stored

aliases = ["best", "epoch_10"]
run.log_artifact(artifact, aliases=aliases)
```

カスタム エイリアス の作成方法については、[カスタム エイリアス を作成する]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}})を参照してください。

出力 Artifacts は、任意の頻度（たとえば、エポックごと、500ステップごとなど）でログに記録でき、自動的に バージョン 管理されます。

#### 学習済み モデル または データセット をログに記録および追跡する

学習済み モデル や データセット など、トレーニングへの入力として使用される Artifacts をログに記録できます。次のスニペットは、Artifact をログに記録し、上記のグラフに示すように、実行中の Run に入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### Artifact をダウンロードする

Artifact（データセット、モデルなど）を再利用すると、`wandb` はローカルにコピーをダウンロード（およびキャッシュ）します。

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションにあり、自動的に生成される エイリアス （`latest`、`v2`、`v3`）またはログ記録時に手動で生成される エイリアス （`best_accuracy` など）で参照できます。

（`wandb.init` を介して）`wandb` Run を作成せずに Artifact をダウンロードするには（たとえば、分散環境または単純な推論の場合）、代わりに[wandb API]({{< relref path="/ref/python/public-api" lang="ja" >}})で Artifact を参照できます。

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Artifacts のダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

### ハイパーパラメーター を チューニング する

ライブラリで W&B ハイパーパラメーター チューニング 、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}})を活用したい場合は、ライブラリに追加することもできます。

### 高度な インテグレーション

高度な W&B インテグレーション がどのようなものかについては、次の インテグレーション を参照してください。ほとんどの インテグレーション はこれほど複雑ではありません。

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)
