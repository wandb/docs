---
title: どの ライブラリ にも wandb を追加する
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を追加する

このガイドでは、あなたの Python ライブラリに W&B を組み込むためのベストプラクティスを紹介します。これにより、強力な 実験管理、GPU とシステム監視、モデルのチェックポイント保存 などを自分のライブラリで利用できます。

{{% alert %}}
まだ W&B の使い方を学習中であれば、読み進める前に [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) など他の W&B ガイドを参照することをおすすめします。
{{% /alert %}}

ここでは、単一の Python トレーニングスクリプトや Jupyter ノートブックよりも複雑なコードベースで作業する場合の、役立つヒントやベストプラクティスを解説します。扱うトピックは次のとおりです。

* セットアップ要件
* ユーザーのログイン
* wandb の Run を開始する
* Run の config を定義する
* W&B へログを送る
* 分散トレーニング
* モデルのチェックポイント保存 など
* ハイパーパラメータのチューニング
* 高度なインテグレーション

### セットアップ要件

始める前に、あなたのライブラリの依存関係として W&B を必須にするかどうかを決めましょう。

#### インストール時に W&B を必須にする

W&B の Python ライブラリ（`wandb`）を依存関係ファイルに追加します。たとえば `requirements.txt` に次のように記述します。

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK（`wandb`）をオプションにする方法は 2 つあります。

A. ユーザーが `wandb` を手動インストールせずに `wandb` の機能を使おうとしたときにエラーを投げ、適切なエラーメッセージを表示する:

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    ) 
```

B. Python パッケージを作っている場合は、`pyproject.toml` のオプション依存として `wandb` を追加する:

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

#### APIキー を作成する

APIキー はクライアントやマシンを W&B に認証するためのものです。ユーザープロフィールから APIキー を生成できます。

{{% alert %}}
よりスムーズに行うには、[W&B authorization page](https://wandb.ai/authorize) にアクセスして APIキー を生成する方法がおすすめです。表示された APIキー をコピーして、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上のユーザープロフィールアイコンをクリックします。
1. その後、 **User Settings** を選択し、 **API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された APIキー をコピーします。APIキー を非表示にするにはページを再読み込みします。

#### `wandb` ライブラリをインストールしてログインする

ローカルに `wandb` をインストールしてログインするには:

{{< tabpane text=true >}}
{{% tab header="コマンドライン" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) に APIキー を設定します。

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

上記の手順を踏まずに初めて wandb を使うユーザーでも、スクリプトが `wandb.init` を呼び出すと自動的にログインを促されます。

### Run を開始する

W&B の Run は、W&B によってログされる計算の単位です。通常、1 回のトレーニング実験につき 1 つの W&B Run を対応づけます。

W&B を初期化して Run を開始するには、次のように記述します。

```python
run = wandb.init()
```

任意で Project 名を指定したり、あるいはユーザーに `wandb_project` のようなパラメータで設定してもらうこともできます。エンティティ（ユーザー名やチーム名）も `wandb_entity` のように指定します（`entity` パラメータ）。

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run を終了するには `run.finish()` を呼び出す必要があります。インテグレーションの設計に合う場合は、Run をコンテキストマネージャーとして使いましょう。

```python
# このブロックを抜けると、自動的に run.finish() が呼ばれます。
# 例外で抜けた場合は run.finish(exit_code=1) が使われ、
# Run は失敗としてマークされます。
with wandb.init() as run:
    ...
```

#### `wandb.init` をいつ呼ぶべき？

ライブラリは可能な限り早いタイミングで W&B の Run を作成してください。コンソール出力（エラーメッセージを含む）は W&B の Run の一部としてログされるため、デバッグが容易になります。

#### `wandb` をオプション依存にする

ユーザーがあなたのライブラリを使うときに `wandb` をオプションにしたい場合は、次のいずれかの方法が使えます。

* `wandb` フラグを定義する（例）:

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

* あるいは、`wandb.init` で `wandb` を `disabled` に設定する:

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

* あるいは、`wandb` をオフラインに設定する（注意: これは `wandb` 自体は実行しますが、インターネット経由で W&B と通信しようとはしません）:

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
`wandb` の run config を使うと、W&B の Run 作成時にモデルやデータセットなどのメタデータを付与できます。この情報は、実験を比較し、主な違いを素早く理解するのに役立ちます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs テーブル" >}}

ログできる一般的な config パラメータの例:

* モデル名、バージョン、アーキテクチャーのパラメータなど
* データセット名、バージョン、学習/検証のサンプル数など
* 学習率、バッチサイズ、オプティマイザー などのトレーニングパラメータ

次のコードスニペットは config をログする方法を示しています。

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run の config を更新する
`wandb.Run.config.update` を使って config を更新できます。辞書を後から更新できると、辞書定義後に得られるパラメータ（例: モデルのインスタンス化後に取得したパラメータ）を追加するのに便利です。

```python
run.config.update({"model_parameters": 3500})
```

config ファイルの定義方法については、[実験を設定する]({{< relref path="/guides/models/track/config" lang="ja" >}}) を参照してください。

### W&B にログを送る

#### メトリクスをログする

キーをメトリクス名とする辞書を作り、その辞書オブジェクトを [`run.log`]({{< relref path="/guides/models/track/log" lang="ja" >}}) に渡します。

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

メトリクスが多い場合は、`train/...` や `val/...` のようにメトリクス名にプレフィックスを付けることで、UI 上で自動的にグルーピングできます。これにより、W&B の Workspace で学習と検証のメトリクス（または分けたい他の種類のメトリクス）が別セクションとして表示されます。

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

[`wandb.Run.log()` のリファレンス]({{< relref path="/guides/models/track/log" lang="ja" >}}) も参照してください。

#### x 軸のずれを防ぐ

同じトレーニングステップで `run.log` を複数回呼ぶと、wandb SDK は `run.log` の呼び出しごとに内部ステップカウンタをインクリメントします。このカウンタは、トレーニングループ内のステップと一致しない場合があります。

この状況を避けるには、`wandb.init` の直後に一度だけ `run.define_metric` を使って x 軸のステップを明示的に定義します。

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

グロブパターン `*` は、すべてのメトリクスがチャートの x 軸として `global_step` を使うことを意味します。特定のメトリクスだけを `global_step` に紐づけたい場合は、次のようにメトリクス名を指定できます。

```python
run.define_metric("train/loss", step_metric="global_step")
```

そして、`run.log` を呼ぶたびにメトリクス、`step` 用のメトリクス、`global_step` を一緒にログします。

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合（例: 検証ループ中は "global_step" が利用できない等）は、直前にログされた "global_step" の値が wandb によって自動的に使用されます。この場合、必要な時点で値が定義済みになるように、該当メトリクスの初期値を先にログしておいてください。

#### 画像、テーブル、音声などをログする

メトリクスに加えて、プロット、ヒストグラム、テーブル、テキスト、画像・動画・音声・3D などのメディアもログできます。

データをログする際の考慮点:

* メトリクスはどれくらいの頻度でログすべきか？オプションにすべきか？
* どの種類のデータが可視化に有用か？
  * 画像なら、サンプル予測やセグメンテーションマスクなどをログして、時間とともにどう変化するかを確認できます。
  * テキストなら、後から探索できるようにサンプル予測のテーブルをログできます。

メディア、オブジェクト、プロットなどの詳細は [ログのガイド]({{< relref path="/guides/models/track/log" lang="ja" >}}) を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークでは、次のいずれかのワークフローを採用できます。

* どのプロセスが「main」かを判定し、そのプロセスでのみ `wandb` を使います。他プロセスから必要なデータは、まず main プロセスに集約します。（このワークフローを推奨）
* すべてのプロセスで `wandb` を呼び出し、同じ一意の `group` 名を付けて自動的にグルーピングします。

詳細は [Log Distributed Training Experiments]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。

### モデルのチェックポイントやその他をログする

フレームワークがモデルやデータセットを使用・生成する場合は、それらをログして完全な追跡可能性を確保できます。さらに、W&B の Artifacts を通じて、パイプライン全体を wandb が自動的にモニタリングできます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="W&B に保存された Datasets と Model Checkpoints" >}}

Artifacts を使う際、次のような設定をユーザーに任意で指定してもらうと便利な場合があります（必須ではありません）。

* モデルのチェックポイントやデータセットをログするかどうか（オプションにしたい場合）
* 入力として使用する Artifact のパス/参照（例: `user/project/artifact`）
* Artifacts をログする頻度

#### モデルのチェックポイントをログする

モデルのチェックポイントを W&B にログできます。Run ごとに出力を区別するため、一意な `wandb` の Run ID を活用して出力モデルのチェックポイント名を付けると便利です。また、有用なメタデータや、モデルごとのエイリアスも以下のように追加できます。

```python
metadata = {"eval/accuracy": 0.8, "train/steps": 800} 

artifact = wandb.Artifact(
                name=f"model-{run.id}", 
                metadata=metadata, 
                type="model"
                ) 
artifact.add_dir("output_model") # モデルの重みが保存されているローカルディレクトリー

aliases = ["best", "epoch_10"] 
run.log_artifact(artifact, aliases=aliases)
```

カスタムエイリアスの作成方法は [Create a Custom Alias]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}}) を参照してください。

出力の Artifacts は任意の頻度（例: 各エポック、500 ステップごと等）でログでき、バージョニングは自動で行われます。

#### 学習済みモデルやデータセットをログ・追跡する

トレーニングの入力として使う学習済みモデルやデータセットなどの artifacts もログできます。次のスニペットは Artifact をログし、上のグラフに示したように進行中の Run の入力として追加する例です。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### Artifact をダウンロードする

既存の Artifact（データセット、モデル等）を再利用すると、`wandb` がローカルにコピーをダウンロード（かつキャッシュ）します。

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションで確認でき、エイリアスは自動（`latest`, `v2`, `v3`）またはログ時に手動（`best_accuracy` など）で付与できます。

`wandb` の Run（`wandb.init`）を作成せずに Artifact をダウンロードしたい場合（分散環境や単純な推論など）には、代わりに [wandb API]({{< relref path="/ref/python/public-api/index.md" lang="ja" >}}) で Artifact を参照できます。

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細は [Download and Use Artifacts]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}}) を参照してください。

### ハイパーパラメータをチューニングする

W&B のハイパーパラメータチューニングを利用したい場合は、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) をライブラリに組み込むこともできます。

### 高度なインテグレーション

以下のようなインテグレーションでは、より高度な W&B 連携の例を見ることができます。多くのインテグレーションはここまで複雑ではありません。

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)