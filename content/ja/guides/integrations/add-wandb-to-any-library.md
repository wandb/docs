---
title: どのライブラリにも wandb を追加
menu:
  default:
    identifier: add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を追加する

このガイドでは、ご自身の Python ライブラリに W&B をインテグレートし、強力な実験管理、GPU やシステムのモニタリング、モデルのチェックポイント管理などを実現するためのベストプラクティスを紹介します。

{{% alert %}}
W&B の使い方をまだ学習中の方は、まず [Experiment Tracking]({{< relref "/guides/models/track" >}}) など他の W&B ガイドをご覧いただくことをおすすめします。
{{% /alert %}}

ここでは、単一の Python トレーニングスクリプトや Jupyter ノートブックよりも複雑なコードベースで作業する場合のベストなコツやベストプラクティスについて説明します。主なトピックは以下の通りです。

* セットアップ要件
* ユーザーのログイン
* wandb Run の開始
* Run Config の定義
* W&B へのログ
* 分散トレーニング
* モデルチェックポイント管理 など
* ハイパーパラメーターチューニング
* さらに高度なインテグレーション

### セットアップ要件

始める前に、ご自身のライブラリの依存関係として W&B を必須にするかどうかを決めましょう。

#### インストール時に W&B を必須にする

W&B Python ライブラリ（`wandb`）を依存ファイル、例として `requirements.txt` に追加してください。

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションとする

W&B SDK（`wandb`）をオプションにするには、主に2つの方法があります。

A. W&B の機能をインストール無しで使おうとした際にエラーを発生させ、分かりやすいエラーメッセージを表示する:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    ) 
```

B. Python パッケージを構築する場合は、`pyproject.toml` ファイルに `wandb` をオプション依存として追加する:

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

### ユーザーのログイン

#### APIキーの作成

APIキーは W&B へのクライアントもしくはマシンの認証に使います。APIキーはユーザープロフィールから発行できます。

{{% alert %}}
より簡単な方法として、[W&B の認証ページ](https://wandb.ai/authorize)に直接アクセスして APIキーを生成することもできます。表示された APIキーをコピーしてパスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 画面右上のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選び、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックし、表示された APIキーをコピーします。再び APIキーを隠すにはページをリロードしてください。

#### `wandb` ライブラリのインストールとログイン

`wandb` ライブラリをローカルにインストールし、ログインするには:

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. [環境変数]({{< relref "/guides/models/track/environment-variables.md" >}}) `WANDB_API_KEY` にご自身の APIキーを設定します。

    ```bash
    export WANDB_API_KEY=<your_api_key>
    ```

1. `wandb` ライブラリをインストールし、ログインします。



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

ユーザーが上記のいずれの手順も踏まずに初めて wandb を利用する場合、スクリプト内で `wandb.init` が呼ばれたタイミングで自動的にログインを促されます。

### Run の開始

W&B Run は W&B に記録される計算単位です。通常、1つのトレーニング実験ごとに1つの Run です。

W&B を初期化し、Run を開始するには以下のコードを利用します。

```python
run = wandb.init()
```

プロジェクト名やユーザー名・チーム名（エンティティ）などをパラメータとして指定したい場合は以下のようにします。

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run を終了するには必ず `run.finish()` を呼ぶ必要があります。もしインテグレーションの設計上許されるのであれば、Run をコンテキストマネージャとして利用するのがおすすめです。

```python
# このブロックから抜けると自動的に run.finish() が呼ばれます。
# 例外で抜けた場合も run.finish(exit_code=1) となり、
# Runは失敗とマークされます。
with wandb.init() as run:
    ...
```

#### `wandb.init` を呼ぶタイミング

あなたのライブラリでは、できるだけ早いタイミングで W&B Run を作成しておくのが望ましいです。コンソールへの出力（エラーメッセージ含む）が W&B Run に記録されるためデバッグが簡単になります。

#### `wandb` をオプション依存にする

利用者がライブラリ使用時に `wandb` をオプションにしたい場合、以下のいずれかの方法を推奨します。

* `wandb` 用のフラグを定義する例:

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

* または、`wandb.init` の mode を `disabled` にセットする:

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

* または、`wandb` をオフラインに設定することで、W&B には通信せずローカル実行する方法もあります ※この場合も wandb の機能自体は使えます:

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
`wandb` の Run config を使うことで、モデルやデータセットに関するメタデータを Run 作成時に付与できます。これを利用すると、異なる実験の比較や主な違いを迅速に把握できます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs table" >}}

一般的なログ対象パラメータ例:

* モデル名・バージョン・アーキテクチャー関連パラメータなど
* データセット名・バージョン・学習/バリデーションのサンプル数など
* 学習率やバッチサイズ・オプティマイザー等のトレーニングパラメータ

以下のコードスニペットのように config を記録できます。

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run config の更新
`wandb.Run.config.update` を使うことで config を更新できます。これはパラメータが辞書定義後に決定される場合に有用です。例えば、モデルのインスタンス化後にモデルパラメータを追加したい場合などです。

```python
run.config.update({"model_parameters": 3500})
```

config ファイルの詳細は [実験の設定]({{< relref "/guides/models/track/config" >}}) をご参照ください。

### W&B へのログ

#### メトリクスのログ

メトリクス名をキーとした辞書を作成し、このオブジェクトを[`run.log`]({{< relref "/guides/models/track/log" >}}) に渡します。

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

メトリクスが多い場合はメトリクス名にプレフィックス（例: `train/...`, `val/...`）を付けることで UI 上で自動的にグルーピングできます。これによりトレーニング・バリデーション・その他のメトリクスを W&B Workspace 内で分けて表示できます。

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

[`wandb.Run.log()` のリファレンス]({{< relref "/guides/models/track/log" >}}) もご確認ください。

#### x軸のずれ（step のミスマッチ）を防ぐ

同じトレーニングステップで複数回 `run.log` を呼ぶと、wandb SDK の内部ステップカウンターがその都度増加し、トレーニングループの step とずれることがあります。

この状況を避けるには、`wandb.init` の直後に `run.define_metric` を使って、x軸用の step を明示的に指定してください:

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

glob パターン `*` は、全てのメトリクスが `global_step` を x軸として使うことを意味します。特定のメトリクスだけ `global_step` と紐付けたい場合は下記のようにします。

```python
run.define_metric("train/loss", step_metric="global_step")
```

以後、run.log を呼ぶ際には `step` メトリクス・`global_step` も一緒にログします。

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

もし独立したステップ変数が利用できない場合（例: 検証ループで `global_step` が取得できない）、wandb は直前にログされた値を自動利用します。この場合、必要な時にメトリクスが定義済みとなるよう最初に値を必ずログしておいてください。

#### 画像、テーブル、音声などのログ

メトリクス以外にも、プロット・ヒストグラム・テーブル・テキスト・画像・動画・音声・3D など様々なメディアをログできます。

データログ時のポイント:

* どのくらいの頻度でメトリクスをログするか？オプションとするべきか？
* どんな種類のデータが可視化に役立つか？
  * 画像なら、予測例・セグメンテーションマスク等をログして時間経過をチェック
  * テキストなら、予測サンプルのテーブルを記録し後から探索可能

メディア・オブジェクト・プロット等の詳細は [ログガイド]({{< relref "/guides/models/track/log" >}}) を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークの場合、以下いずれかのワークフローを採用できます。

* 「メイン」プロセスを検出し、`wandb` はそこでのみ利用。他プロセスの必要なデータもメイン経由で処理（この方法を推奨）
* 全プロセスで `wandb` を実行し、同じ一意な `group` 名で自動グルーピング

詳細は [分散トレーニング実験のログ]({{< relref "/guides/models/track/log/distributed-training.md" >}}) をご参照ください。

### モデルのチェックポイントなどをログする

ご自身のフレームワークがモデルやデータセットを使う場合、それらをログし、wandb でパイプライン全体を W&B Artifacts で監視できます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="W&B に保存された Datasets と Model Checkpoints" >}}

Artifacts 利用時、ユーザーに以下を指定できるようにしておくと便利です（必須ではありません）。

* モデルチェックポイントやデータセットのログの可否（オプションにしたい場合）
* 入力として使っている artifact のパス/参照（例: `user/project/artifact` など）
* Artifacts をログする頻度

#### モデルチェックポイントのログ

モデルチェックポイントを W&B へ記録可能です。各 Run のユニークな `wandb` ID を利用して、出力ファイル名を Run 単位で分かりやすくできます。また、メタデータやエイリアスの追加も可能です。

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # モデルの重みが保存されているローカルディレクトリ

aliases = ["best", "epoch_10"] 
run.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアス作成方法の詳細は [Create a Custom Alias]({{< relref "/guides/core/artifacts/create-a-custom-alias/" >}}) をご参照ください。

Artifacts の出力頻度は自由に設定できます（例: 各エポック毎、500ステップ毎等）。バージョン管理も自動です。

#### 学習済みモデルやデータセットのログ・トラッキング

入力として使うアーティファクト（学習済みモデルやデータセットなど）もログできます。下記スニペットは、Artifact をログし、進行中の Run への入力として追加する例です。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### Artifact のダウンロード

Artifact（データセット・モデル等）の再利用やローカルへのダウンロード（キャッシュ込み）は下記の通り:

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションで一覧表示でき、`latest`、`v2`、`v3` などの自動エイリアス、またはログ時に付けたエイリアス（`best_accuracy`等）で参照できます。

`wandb` Run を作成せずに Artifact をダウンロードしたい場合（分散環境や単純な推論等）は、[wandb API]({{< relref "/ref/python/public-api/index.md" >}}) で以下のように取得可能です。

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細は [Download and Use Artifacts]({{< relref "/guides/core/artifacts/download-and-use-an-artifact" >}}) をご参照ください。

### ハイパーパラメーターチューニング

ライブラリで W&B のハイパーパラメーターチューニングを活用したい場合は、[W&B Sweeps]({{< relref "/guides/models/sweeps/" >}}) を追加できます。

### 高度なインテグレーション

さらに高度な W&B インテグレーション例については、以下の連携例をご覧ください。ほとんどのインテグレーションはここまで複雑である必要はありません。

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)
