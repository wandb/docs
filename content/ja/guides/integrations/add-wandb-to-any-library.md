---
title: どのライブラリにも wandb を追加する
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を導入する

このガイドでは、ご自身の Python ライブラリに W&B を組み込んで、強力な実験管理、GPU やシステムのモニタリング、モデルのチェックポイント保存などを実現するためのベストプラクティスをご紹介します。

{{% alert %}}
もしまだ W&B の使い方を学習中であれば、まずはこのドキュメント内の [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) などの他の W&B ガイドをご覧いただくことをおすすめします。
{{% /alert %}}

ここでは、単一の Python トレーニングスクリプトや Jupyter ノートブックよりも複雑なコードベースを扱う場合のベストな方法やポイントを解説します。以下のトピックを取り上げます。

* セットアップ要件
* ユーザーのログイン
* wandb Run の開始
* Run Config の定義
* W&B へのログ
* 分散トレーニング
* モデルのチェックポイント保存など
* ハイパーパラメータチューニング
* 高度なインテグレーション

### セットアップ要件

まずは、あなたのライブラリに W&B をインストール時点で必須とするか、オプションにするかを決めてください。

#### インストール時に W&B を必須にする

W&B の Python ライブラリ（`wandb`）を依存ファイルに追加します。例として `requirements.txt` ファイルの場合：

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK（`wandb`）をオプションとしたい場合、次のいずれかの方法が使えます。

A. `wandb` がインストールされていない状態で W&B の機能を使おうとしたユーザーに、適切なエラーメッセージを出してエラーを発生させる：

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "wandb を使おうとしていますが、現在インストールされていません。"
        "pip install wandb でインストールしてください"
    ) 
```

B. Python パッケージをビルドしている場合、`pyproject.toml` ファイルの optional dependency に `wandb` を追加：

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

#### API キーの作成

API キーは、クライアントやマシンを W&B へ認証するものです。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
より簡単に API キーを作成したい場合は、[W&B の認証ページ](https://wandb.ai/authorize) から直接生成が可能です。表示された API キーをコピーして、パスワードマネージャなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザーアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックして表示された API キーをコピーします。隠したい場合はページをリロードしてください。

#### `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールし、ログインします：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. ご自身の API キーを `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}})として設定します。

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

{{% tab header="Python notebook" value="python-notebook" %}}

```notebook
!pip install wandb

import wandb
wandb.login()
```

{{% /tab %}}
{{< /tabpane >}}

はじめて wandb を使うユーザーが上記いずれの手順も実行せずに script 内で `wandb.init` を呼び出すと、自動的にログインを促されます。

### Run の開始

W&B Run とは、W&B によって記録される計算実行の単位です。通常、1つのトレーニング実験につき1つの Run を紐付けます。

W&B を初期化し、Run を開始するには次のようにします：

```python
run = wandb.init()
```

プロジェクト名なども指定できます。また、ユーザーやチーム名を entity パラメータとして扱うこともできます（例：`wandb_project` や `wandb_entity`）:

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run を終了するには必ず `run.finish()` を呼び出してください。この処理がインテグレーションの設計に合う場合は、Run をコンテキストマネージャとして使うと便利です：

```python
# このブロックを抜けると自動で run.finish() が呼ばれます。
# 例外でブロックを抜けた場合は run.finish(exit_code=1) が使われ、
# その Run が失敗扱いとなります。
with wandb.init() as run:
    ...
```

#### `wandb.init` をいつ呼ぶべきか？

W&B Run の生成はできるだけ早い段階で行いましょう。というのも、コンソールに出力される内容（エラーメッセージも含む）はすべて W&B Run の一部として記録されるためです。これによってデバッグも容易になります。

#### `wandb` をオプション依存にする方法

利用者が必ずしも `wandb` を必要としない場合、次のようなフラグを設けて切り替えたり：

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

あるいは、`wandb.init` の `mode` を `disabled` にすることでログ送信を無効化できます:

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

もしくは

```bash
wandb disabled
```
{{% /tab %}}

{{< /tabpane >}}

もしくは W&B をオフラインモードで使う方法もあります。この場合も wandb は動作しますが、インターネットを通じて W&B サーバーとは通信しません：

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

### Run config の定義
`wandb` の run config を使うことで、モデルやデータセットなど Run 作成時のメタデータを付加できます。これらの情報は実験ごとの差分を比較したり、主要な違いをすぐに理解するのに役立ちます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

典型的な config パラメータの例：

* モデル名、バージョン、アーキテクチャパラメータ など
* データセット名、バージョン、学習/検証サンプル数 など
* トレーニングパラメータ（学習率、バッチサイズ、オプティマイザーなど）

config をログする例：

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run config の更新
`wandb.Run.config.update` を使って config を後から更新することも可能です。たとえばモデルのインスタンス生成後に、そのパラメータを追加したい場合などに便利です。

```python
run.config.update({"model_parameters": 3500})
```

config ファイルの詳細については [実験の設定]({{< relref path="/guides/models/track/config" lang="ja" >}}) をご覧ください。

### W&B へのログ

#### メトリクスのログ

辞書型で各メトリクスのキーと値を設定し、それを [`run.log`]({{< relref path="/guides/models/track/log" lang="ja" >}}) に渡します：

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

多くのメトリクスを記録する場合は、メトリクス名にプレフィックス（例：`train/...` や `val/...`）を使うことで UI 上でグルーピングできます。これによりトレーニングや検証、その他区分で W&B Workspace 内のメトリクスが整理されます。

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="W&B Workspace" >}}

[`wandb.Run.log()` のリファレンス]({{< relref path="/guides/models/track/log" lang="ja" >}})も合わせてご覧ください。

#### x軸のズレを防ぐ

同じトレーニングステップで `run.log` を複数回呼ぶ場合、wandb SDK の内部カウンターが自動でインクリメントされ、実際のトレーニングループのステップとズレる可能性があります。

この問題を避けるには、`wandb.init` 呼び出し直後に `run.define_metric` で x 軸となるステップを明示的に定義してください：

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

`*` はすべてのメトリクスが `global_step` を x 軸に使うことを意味します。特定のメトリクスのみ `global_step` でログしたい場合はこのように指定します：

```python
run.define_metric("train/loss", step_metric="global_step")
```

その後は、`run.log` を呼ぶたびにメトリクスと一緒に `global_step` をログします：

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

もし独立した step 変数（例えば検証ループ時の "global_step" など）が利用できない場合でも、wandb は直前に記録した値を自動的に参照します。そのため、最初のメトリクス値を必ずログして定義しておきましょう。

#### 画像、テーブル、音声などをログする

メトリクス以外にも、プロット・ヒストグラム・テーブル・テキスト・画像・動画・音声・3D など多様なデータをログできます。

データを記録する際に考慮するポイント：

* どの頻度でログするか？ オプションにすべきか？
* どんなデータが可視化に役立つか？
  * 画像なら、サンプル予測やセグメンテーションマスクなど、経時変化の見える化
  * テキストなら、サンプル予測テーブルによる探索性の向上

詳細は [ロギングガイド]({{< relref path="/guides/models/track/log" lang="ja" >}}) をご覧ください。

### 分散トレーニング

分散環境をサポートするフレームワークでは、下記のいずれかのワークフローを適用できます。

* 「メインプロセス」を検出し、`wandb` はそこでのみ使用。他のプロセスから必要なデータをメインへ受け渡して記録する（推奨）。
* すべてのプロセスで `wandb` を呼び、同じ `group` 名で自動グルーピング。

より詳しくは、[分散トレーニング実験のロギング]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}})をご覧ください。

### モデルのチェックポイント保存など

フレームワークがモデルやデータセットを利用したり生成したりする場合、wandb でそれらも一緒に記録すれば、W&B Artifacts を通してパイプライン全体を追跡できます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="Stored Datasets and Model Checkpoints in W&B" >}}

Artifacts を使う際、ユーザーに次のような項目を定義させられると便利な場合があります（必須ではありません）：

* モデルのチェックポイントやデータセットのログの有無（オプションにしたい場合）
* 入力として使用する artifact のパス/参照（例：`user/project/artifact`）
* Artifacts の記録頻度

#### モデルチェックポイントの保存

モデルチェックポイントも W&B へ記録可能です。Run ごとに一意な `wandb` の Run ID をファイル名等に使えば、Run 間での区別がしやすくなります。また、メタデータやエイリアスの追加もできます。

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # モデルの重みが格納されているローカルディレクトリ

aliases = ["best", "epoch_10"] 
run.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアスの作成例は [カスタムエイリアスを作成する]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}})をご覧ください。

Artifacts の出力タイミング（例：各エポックごと、500ステップごと等）は自由で、自動でバージョン管理されます。

#### 学習済みモデルやデータセットの記録と追跡

学習に使う入力（学習済みモデルやデータセットなど）を artifact として記録し、グラフ内で ongoing Run に入力として追加できます。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### artifact のダウンロード

artifact（データセット・モデル等）の再利用時に、`wandb` がローカルにダウンロード（キャッシュ）します：

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションで確認でき、`latest`、`v2`、`v3` や `best_accuracy` などのエイリアスで参照可能です。

また、推論時や分散環境で `wandb` run を作成せず artifact だけダウンロードしたい場合は [wandb API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) を使います：

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細情報は [Artifacts のダウンロードと利用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}}) をご覧ください。

### ハイパーパラメータのチューニング

W&B のハイパーパラメータチューニングを活用したい場合は、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) をライブラリに組み込むこともできます。

### 高度なインテグレーション例

より高度な W&B インテグレーションの事例は下記をご覧ください。大半のケースではここまで複雑なものにはなりません。

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)