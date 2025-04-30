---
title: 任意のライブラリに wandb を追加する
menu:
  default:
    identifier: ja-guides-integrations-add-wandb-to-any-library
    parent: integrations
weight: 10
---

## 任意のライブラリに wandb を追加する

このガイドでは、独自の Python ライブラリに W&B をインテグレーションするためのベストプラクティスを提供します。このことで、強力な実験管理、GPU およびシステム監視、モデルチェックポイントなどが利用可能になります。

{{% alert %}}
W&B の使い方をまだ学んでいる場合は、読み進める前に [実験管理]({{< relref path="/guides/models/track" lang="ja" >}}) など、他の W&B ガイドを探索することをお勧めします。
{{% /alert %}}

ここでは、作業しているコードベースが単一の Python トレーニングスクリプトや Jupyter ノートブックよりも複雑な場合のベストなヒントとベストプラクティスを紹介します。対象となるトピックは以下の通りです：

* 設定要件
* ユーザーログイン
* wandb Run の開始
* Run Config の定義
* W&B へのログ記録
* 分散トレーニング
* モデルチェックポイントなど
* ハイパーパラメータチューニング
* 高度なインテグレーション

### 設定要件

始める前に、ライブラリの依存関係に W&B を必須にするかどうかを決めてください：

#### インストール時に W&B を必須にする

W&B の Python ライブラリ（`wandb`）を `requirements.txt` ファイルなどに含めて依存関係ファイルに追加します：

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時に W&B をオプションにする

W&B SDK（`wandb`）をオプションにする方法は2つあります：

A. ユーザーが手動でインストールせずに `wandb` 機能を使用しようとしたときにエラーメッセージを表示してエラーを発生させる：

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        "You are trying to use wandb which is not currently installed."
        "Please install it using pip install wandb"
    ) 
```

B. Python パッケージをビルドする場合は `wandb` をオプションの依存関係として `pyproject.toml` ファイルに追加する：

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

API キーはクライアントまたはマシンを W&B に認証するものです。ユーザープロフィールから API キーを生成できます。

{{% alert %}}
よりスムーズな方法として、[https://wandb.ai/authorize](https://wandb.ai/authorize) に直接アクセスして API キーを生成できます。表示された API キーをコピーし、パスワードマネージャーなどの安全な場所に保存してください。
{{% /alert %}}

1. 右上隅のユーザープロフィールアイコンをクリックします。
1. **User Settings** を選択し、**API Keys** セクションまでスクロールします。
1. **Reveal** をクリックします。表示された API キーをコピーします。API キーを非表示にするには、ページをリロードします。

#### `wandb` ライブラリのインストールとログイン

ローカルに `wandb` ライブラリをインストールしてログインします：

{{< tabpane text=true >}}
{{% tab header="Command Line" value="cli" %}}

1. `WANDB_API_KEY` [環境変数]({{< relref path="/guides/models/track/environment-variables.md" lang="ja" >}}) にあなたの API キーを設定します。

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

ユーザーが上記のいずれの手順も行わずに初めて wandb を使用する場合、あなたのスクリプトが `wandb.init` を呼び出す際に自動的にログインを求められます。

### Run の開始

W&B Run は、W&B によってログ記録された計算の単位です。通常、トレーニング実験ごとに1つの W&B Run を関連付けます。

以下のコードで W&B を初期化して Run を開始します：

```python
run = wandb.init()
```

オプションとして、プロジェクトの名前をつけることができます。また、コード内で `wandb_project` といったパラメータを使ってユーザーに設定してもらうこともできます。エンティティのパラメータについては `wandb_entity` などのユーザー名やチーム名を使用します：

```python
run = wandb.init(project=wandb_project, entity=wandb_entity)
```

Run を終了するには `run.finish()` を呼び出す必要があります。次のように Run をコンテキストマネージャとして使うこともできます：

```python
# このブロックが終了すると、自動的に run.finish() が呼び出されます。
# 例外によって終了した場合、run.finish(exit_code=1) を使用して
# Run を失敗とマークします。
with wandb.init() as run:
    ...
```

#### `wandb.init` を呼び出すタイミング？

ライブラリは、W&B Run を可能な限り早く作成するべきです。なぜなら、コンソール出力に含まれるエラーメッセージなどの内容が W&B Run の一部としてログされ、デバッグが容易になるからです。

#### `wandb` をオプション依存関係として使用する

ユーザーがライブラリを使うときに `wandb` をオプションにしたい場合、以下のいずれかの方法を使用できます：

* 次のように `wandb` フラグを定義する：

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

* または、`wandb.init` で `wandb` を `disabled` に設定する：

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

* または、`wandb` をオフラインに設定します - これは wandb を実行はしますが、インターネットを介して W&B に通信を試みません：

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

### Run Config の定義

`wandb` の Run Config を使って、W&B Run を作成する際にモデルやデータセットに関するメタデータを提供できます。この情報を利用して、異なる実験を比較し、その主な違いをすばやく理解することができます。

{{< img src="/images/integrations/integrations_add_any_lib_runs_page.png" alt="W&B Runs テーブル" >}}

ログ可能な一般的な設定パラメータには以下が含まれます：

* モデル名、バージョン、アーキテクチャパラメータなど
* データセット名、バージョン、トレイン/バルの例数など
* 学習パラメータ（学習率、バッチサイズ、オプティマイザーなど）

以下のコードスニペットは設定をログする方法を示しています：

```python
config = {"batch_size": 32, ...}
wandb.init(..., config=config)
```

#### Run Config の更新

`run.config.update` を使用して設定を更新します。設定辞書の更新は、辞書が定義された後にパラメータが取得された場合に役立ちます。たとえば、モデルがインスタンス化された後にモデルのパラメータを追加したい場合などです。

```python
run.config.update({"model_parameters": 3500})
```

設定ファイルを定義する方法の詳細については、[実験を設定する]({{< relref path="/guides/models/track/config" lang="ja" >}})を参照してください。

### W&B へのログ記録

#### メトリクスのログ記録

キーがメトリクス名となる辞書を作成し、この辞書オブジェクトを [`run.log`]({{< relref path="/guides/models/track/log" lang="ja" >}}) に渡します：

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { "loss": loss } 
        run.log(metrics)
```

メトリクスが多い場合、メトリクス名にプレフィックスを使用して UI 上で自動的にグループ化することができます。例えば、`train/...` と `val/...` を使用することで、トレーニングや検証メトリクス、その他のメトリクスを分けたセクションが W&B ワークスペースに作られます：

```python
metrics = {
    "train/loss": 0.4,
    "train/learning_rate": 0.4,
    "val/loss": 0.5, 
    "val/accuracy": 0.7
}
run.log(metrics)
```

{{< img src="/images/integrations/integrations_add_any_lib_log.png" alt="2つの別々のセクションがあるW&Bワークスペース" >}}

[`run.log` の詳細を学ぶ]({{< relref path="/guides/models/track/log" lang="ja" >}})。

#### x軸の非整合を防ぐ

同じトレーニングステップで `run.log` を複数回呼び出すと、wandb SDK は `run.log` を呼び出すたびに内部のステップカウンタを増加させます。このカウンタはトレーニングループ内のトレーニングステップと一致しないことがあります。

このような状況を避けるために、`wandb.init` を呼び出した直後に `run.define_metric` を使用して x 軸のステップを明示的に定義してください：

```python
with wandb.init(...) as run:
    run.define_metric("*", step_metric="global_step")
```

グロブパターンの `*` は、すべてのメトリクスがチャートの x 軸として `global_step` を使用することを意味します。特定のメトリクスのみを `global_step` に対してログする場合は、代わりにそれらを指定できます：

```python
run.define_metric("train/loss", step_metric="global_step")
```

その後、メトリクス、`step` メトリクス、および `global_step` を `run.log` を呼び出すたびにログします：

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    run.log({"global_step": step, "train/loss": 0.1})
    run.log({"global_step": step, "eval/loss": 0.2})
```

独立したステップ変数にアクセスできない場合、たとえば「global_step」が検証ループ中に利用できない場合、 wandb は自動的に以前にログされた「global_step」 の値を使用します。この場合、メトリクスの初期値をログして、その値が必要なときに定義されるようにしてください。

#### 画像、テーブル、オーディオなどのログ記録

メトリクスに加えて、プロット、ヒストグラム、テーブル、テキスト、画像、動画、オーディオ、3D などのメディアをログすることができます。

データをログする際の考慮事項には以下が含まれます：

* メトリクスはどのくらいの頻度でログされるべきか？ オプション化すべきか？
* 視覚化に役立つデータの種類は何か？
  * 画像の場合、サンプル予測、セグメンテーションマスクなどのログを記録して、時間の経過を見て進化を追うことができます。
  * テキストの場合、後で検討できるサンプル予測のテーブルをログすることができます。

[メディア、オブジェクト、プロットなどのログ記録について詳しく学ぶ]({{< relref path="/guides/models/track/log" lang="ja" >}})。

### 分散トレーニング

分散環境をサポートするフレームワークでは、以下のワークフローのいずれかを適用することができます：

* 「メイン」のプロセスを検出し、そこでのみ `wandb` を使用してください。他のプロセスから必要なデータは最初にメインプロセスにルーティングされなければなりません（このワークフローが推奨されます）。
* すべてのプロセスで `wandb` を呼び出し、それらすべてに同じ一意の `group` 名を与えて自動グループ化します。

詳細については [分散トレーニング実験のログ記録]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) を参照してください。

### モデルチェックポイントとその他のログ記録

フレームワークがモデルまたはデータセットを使用または生成する場合、W&B Artifacts を通じて wandb で完全なトレース可能性を持ってそれらをログし、パイプライン全体を自動的に監視させることができます。

{{< img src="/images/integrations/integrations_add_any_lib_dag.png" alt="W&B に保存されたデータセットとモデルチェックポイント" >}}

Artifacts を使用しているとき、ユーザーに次のことを定義させることは有用ですが必須ではありません：

* モデルチェックポイントまたはデータセットをログする機能を有すること（任意にする場合）。
* 使用されるアーティファクトのパス/参照を入力として使用する場合。たとえば、`user/project/artifact` のような指定。
* Artifacts をログする頻度。

#### モデルチェックポイントのログ記録

モデルチェックポイントを W&B にログすることができます。ユニークな `wandb` Run ID を使用して出力モデルチェックポイントに名前を付け、Run 間でそれらを区別するのが有効です。また、有用なメタデータを追加することもできます。さらに、以下のようにモデルごとにエイリアスを追加することもできます：

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

カスタムエイリアスの作成方法については [カスタムエイリアスを作成する]({{< relref path="/guides/core/artifacts/create-a-custom-alias/" lang="ja" >}}) を参照してください。

Artifacts は任意の頻度で出力ログが可能（例えば、各エポックごと、各500ステップごとなど）であり、これらは自動的にバージョン管理されます。

#### 学習済みモデルまたはデータセットのログと追跡

トレーニングの入力として使用されるアーティファクト（学習済みモデルやデータセットなど）をログすることができます。以下のスニペットでは、アーティファクトをログし、上記のグラフのように進行中の Run の入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name="flowers", type="dataset")
artifact_input_data.add_file("flowers.npy")
run.use_artifact(artifact_input_data)
```

#### アーティファクトをダウンロードする

アーティファクト（データセット、モデルなど）を再利用する場合、 `wandb` はローカルにコピー（およびキャッシュ）をダウンロードします：

```python
artifact = run.use_artifact("user/project/artifact:latest")
local_path = artifact.download("./tmp")
```

Artifacts は W&B の Artifacts セクションで見つかり、自動で生成されるエイリアス（`latest`, `v2`, `v3`）またはログ時に手動で生成されるエイリアス（`best_accuracy` など）で参照できます。

たとえば分散環境や単純な推論のために `wandb` Run（`wandb.init` を通して）を作成せずに Artifact をダウンロードしたい場合、代わりに [wandb API]({{< relref path="/ref/python/public-api" lang="ja" >}}) を使用してアーティファクトを参照できます：

```python
artifact = wandb.Api().artifact("user/project/artifact:latest")
local_path = artifact.download()
```

詳細については、[Artのダウンロードと使用]({{< relref path="/guides/core/artifacts/download-and-use-an-artifact" lang="ja" >}})を参照してください。

### ハイパーパラメータのチューニング

ライブラリが W&B ハイパーパラメータチューニングを活用したい場合、[W&B Sweeps]({{< relref path="/guides/models/sweeps/" lang="ja" >}}) もライブラリに追加できます。

### 高度なインテグレーション

以下のインテグレーションで高度な W&B インテグレーションがどのように見えるか見ることができます。ただし、ほとんどのインテグレーションはこれほど複雑ではありません：

* [Hugging Face Transformers `WandbCallback`](https://github.com/huggingface/transformers/blob/49629e7ba8ef68476e08b671d6fc71288c2f16f1/src/transformers/integrations.py#L639)
* [PyTorch Lightning `WandbLogger`](https://github.com/Lightning-AI/lightning/blob/18f7f2d3958fb60fcb17b4cb69594530e83c217f/src/pytorch_lightning/loggers/wandb.py#L53)