---
displayed_sidebar: default
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


# Add wandb to Any Library

このガイドでは、強力な実験管理、GPUとシステムモニタリング、モデルのチェックポイント作成などを自分のライブラリに取り入れるためのW&Bのベストプラクティスを提供します。

:::note
まだW&Bの使い方を学んでいる途中の場合は、他のW&Bガイド、例えば [Experiment Tracking](https://docs.wandb.ai/guides/track) を先に読むことをお勧めします。
:::

ここでは、作業しているコードベースが単一のPythonトレーニングスクリプトやJupyterノートブックよりも複雑な場合のベストなヒントとベストプラクティスを紹介します。カバーするトピックは以下の通りです：

* セットアップ要件
* ユーザーログイン
* wandb Runの開始
* Run Configの定義
* W&Bへのログ
* 分散トレーニング
* モデルのチェックポイント作成など
* ハイパーパラメータチューニング
* 高度なインテグレーション



### セットアップ要件

始める前に、ライブラリの依存関係にW&Bを含めるかどうかを決定してください：

#### インストール時にW&Bを必須にする

W&B Pythonライブラリ（`wandb`）を依存関係ファイル、例えば `requirements.txt` に追加します

```python
torch==1.8.0 
...
wandb==0.13.*
```

#### インストール時にW&Bをオプションにする

W&B SDK（`wandb`）をオプションにするには、以下の2つの方法があります：

A. ユーザーが手動でインストールせずに`wandb`機能を使用しようとしたときにエラーメッセージを表示してエラーを発生させる：

```python
try: 
    import wandb 
except ImportError: 
    raise ImportError(
        “You are trying to use wandb which is not currently installed”
        “Please install it using pip install wandb”
    ) 
```

B. Pythonパッケージを作成している場合、`wandb`をオプションの依存関係として `pyproject.toml` ファイルに追加する。

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

ユーザーがW&Bにログインする方法はいくつかあります：

<Tabs
  defaultValue="bash"
  values={[
    {label: 'Bash', value: 'bash'},
    {label: 'Notebook', value: 'notebook'},
    {label: 'Environment Variable', value: 'environment'},
  ]}>
  <TabItem value="bash">
ターミナルでBashコマンドを使ってW&Bにログインします

```bash
wandb login $MY_WANDB_KEY
```
  </TabItem>
  <TabItem value="notebook">
JupyterやColabノートブックでW&Bにログインする場合

```python
import wandb
wandb.login
```
  </TabItem>
  <TabItem value="environment">

APIキーのために[W&B環境変数](../track/environment-variables.md)を設定します

```bash
export WANDB_API_KEY=$YOUR_API_KEY
```

または

```
os.environ['WANDB_API_KEY'] = "abc123..."
```
  </TabItem>
</Tabs>


ユーザーが上述のいずれのステップも実行せずに初めてwandbを使用する場合、スクリプトが `wandb.init` を呼び出すと自動的にログインを促されます。

### wandb Runの開始

W&B RunはW&Bによってログされる計算の単位です。通常、トレーニング実験ごとに1つのW&B Runを関連付けます。

W&Bを初期化し、コード内でRunを開始するには：

```python
wandb.init()
```

オプションとして、プロジェクト名やエンティティパラメータのためのユーザー名やチーム名などを使用して、プロジェクト名を設定することもできます：

```python
wandb.init(project=wandb_project, entity=wandb_entity)
```

#### `wandb.init` の配置場所

ライブラリはW&B Runをできるだけ早期に作成するべきです。コンソールの出力、エラーメッセージを含む全てがW&B Runの一部としてログされるため、デバッグが簡単になります。

#### ライブラリの `wandb` をオプションにする

ユーザーがライブラリを使用するときに`wandb`をオプションにする場合、以下のいずれかの方法があります：

* `wandb` フラグを定義する：

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

* または、`wandb.init` で `wandb` を無効に設定する

<Tabs
  defaultValue="python"
  values={[
    {label: 'Python', value: 'python'},
    {label: 'Bash', value: 'bash'},
  ]}>
  <TabItem value="python">

```python
wandb.init(mode=“disabled”)
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

* または、`wandb` をオフラインに設定する - この場合も `wandb` は実行されますが、インターネットを介してW&Bに通信しようとしません。

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

### wandb Run Configの定義

`wandb` run configを使用して、モデルやデータセットなどに関するメタデータをW&B Runを作成する際に提供できます。この情報を使用して異なる実験を比較し、主な違いを迅速に理解することができます。

![W&B Runs table](/images/integrations/integrations_add_any_lib_runs_page.png)

一般的にログするconfigパラメータには以下が含まれます：

* モデル名、バージョン、アーキテクチャーパラメータなど
* データセット名、バージョン、トレーニング/検証サンプル数など
* トレーニングパラメータ（学習率、バッチサイズ、オプティマイザーなど）

以下のコードスニペットはconfigをログする方法を示しています：

```python
config = {“batch_size”:32, …}
wandb.init(…, config=config)
```

#### wandb configの更新

`wandb.config.update` を使用してconfigを更新します。設定辞書を更新することは、辞書が定義された後にパラメータを取得する場合に便利です。例えば、モデルのパラメータをモデルのインスタンス化後に追加することが考えられます。

```python
wandb.config.update({“model_parameters” = 3500})
```

configファイルの定義方法についての詳細は、[Configure Experiments with wandb.config](https://docs.wandb.ai/guides/track/config) を参照してください。

### W&Bへのログ

#### メトリクスをログする

キーがメトリック名である辞書を作成します。この辞書オブジェクトを[`wandb.log`](https://docs.wandb.ai/guides/track/log) に渡します：

```python
for epoch in range(NUM_EPOCHS):
    for input, ground_truth in data: 
        prediction = model(input) 
        loss = loss_fn(prediction, ground_truth) 
        metrics = { “loss”: loss } 
        wandb.log(metrics)
```

多くのメトリクスがある場合、メトリクス名にプレフィックスを使用してUIで自動的にグループ化できます。例えば、`train/...`と`val/...` です。これにより、トレーニングと検証メトリクス、または他のメトリックタイプを分離してW&Bワークスペースにセクションを作成できます。

```python
metrics = {
    “train/loss”: 0.4,
    “train/learning_rate”: 0.4,
    “val/loss”: 0.5, 
    “val/accuracy”: 0.7
}
wandb.log(metrics)
```

![A W&B Workspace with 2 separate sections](/images/integrations/integrations_add_any_lib_log.png)

`wandb.log` の詳細については、[Log Data with wandb.log](https://docs.wandb.ai/guides/track/log) を参照してください。

#### x軸のずれを防止する

同じトレーニングステップに対して複数回 `wandb.log` を呼び出す必要がある場合があります。wandb SDKには独自の内部ステップカウンタがあり、`wandb.log` が呼び出されるたびに増加します。したがって、wandbログカウンタがトレーニングステップと一致しない可能性があります。

以下の例の最初のパスでは、`train/loss` の内部 `wandb` ステップは0ですが、 `eval/loss` の内部 `wandb` ステップは1になります。次のパスでは、`train/loss` は2になり、 `eval/loss` のwandbステップは3になります。

```python
for input, ground_truth in data:
    ...
    wandb.log(“train/loss”: 0.1)  
    wandb.log(“eval/loss”: 0.2)
```

これを避けるために、x軸ステップを特定することをお勧めします。`wandb.define_metric` を使用してx軸を定義でき、`wandb.init` が呼び出された後にこれを一度だけ行う必要があります：

```
wandb.init(...)
wandb.define_metric("*", step_metric="global_step")
```

グロブパターン、"\*" は、すべてのメトリクスがチャートのx軸に「global_step」を使用することを意味します。特定のメトリクスのみが「global_step」に対してログされるようにする場合は、それらを指定できます：

```
wandb.define_metric("train/loss", step_metric="global_step")
```

`wandb.define_metric` を呼び出した後、`wandb.log` を呼び出すたびに、メトリクスと `step_metric` の「global_step」をログする必要があります：

```python
for step, (input, ground_truth) in enumerate(data):
    ...
    wandb.log({“global_step”: step, “train/loss”: 0.1})
    wandb.log({“global_step”: step, “eval/loss”: 0.2})
```

独立したステップ変数にアクセスできない場合、たとえば検証ループ中に「global_step」が利用できない場合、wandb は自動的に以前にログされた「global_step」の値を使用します。この場合、最初の値をログしておくことが必要です。

#### 画像、テーブル、テキスト、オーディオなどをログする

メトリクスに加えて、プロット、ヒストグラム、テーブル、テキスト、画像、ビデオ、オーディオ、3Dなどのメディアデータをログできます。

データをログする際の考慮事項：

* メトリクスはどのくらいの頻度でログされるべきか？オプションにすべきか？
* どのようなデータが視覚化に役立つか？
  * 画像の場合、サンプル予測、セグメンテーションマスクなど、経時的な変化を確認するためにログします。
  * テキストの場合、後で探索できるようにサンプル予測のテーブルをログします。

メディア、オブジェクト、プロットなどのログに関する完全なガイドについては、[Log Data with wandb.log](https://docs.wandb.ai/guides/track/log) を参照してください。

### 分散トレーニング

分散環境をサポートするフレームワークの場合、以下のワークフローのいずれかを適用できます：

* 「メイン」プロセスを検出し、そこだけで `wandb` を使用します。他のプロセスからの必要なデータはまずメインプロセスにルーティングされなければなりません。（このワークフローが推奨されます）。
* 各プロセスで `wandb` を呼び出し、同じユニークな `group` 名を与えて自動グループ化します

詳細については、[Log Distributed Training Experiments](../track/log/distributed-training.md) を参照してください。

### モデルのチェックポイント作成などのログ

フレームワークがモデルやデータセットを使用または生成する場合、これらをログして完全なトレーサビリティを持ち、W&B Artifacts を通じてパイプライン全体を自動的に監視できます。

![Stored Datasets and Model Checkpoints in W&B](/images/integrations/integrations_add_any_lib_dag.png)

Artifactsを使用する場合、以下をユーザーが定義できるようにすることが有益ですが、必須ではありません：

* モデルのチェックポイントやデータセットのログを取る能力（任意にしたい場合）
* 入力として使用されるアーティファクトのパス/参照。例えば「user/project/artifact」
* Artifactsをログする頻度

#### モデルのチェックポイントをログする

モデルのチェックポイントをW&Bにログできます。ユニークな`wandb` Run IDを活用して、モデルのチェックポイントを区別できます。さらに有用なメタデータを追加することもできます。また、以下のように各モデルにエイリアスを追加することもできます：

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

カスタムエイリアスの作成方法については、[Create a Custom Alias](https://docs.wandb.ai/guides/artifacts/create-a-custom-alias)を参照してください。

出力Artifactsは任意の頻度で (例えば、各エポック、500ステップ毎など) ログでき、自動的にバージョン管理されます。

#### 学習済みモデルやデータセットのログと追跡

学習の入力として使用されるアーティファクト（学習済みモデルやデータセットなど）をログできます。以下のスニペットは、アーティファクトをログし、グラフに示されるように進行中のRunに入力として追加する方法を示しています。

```python
artifact_input_data = wandb.Artifact(name=”flowers”, type=”dataset”)
artifact_input_data.add_file(“flowers.npy”)
wandb.use_artifact(artifact_input_data)
```

#### W&Bアーティファクトをダウンロードする

アーティファクト（データセット、モデルなど）を再利用し、`wandb` がローカルにコピーをダウンロード（およびキャッシュ）します：

```python
artifact = wandb.run.use_artifact(“user/project/artifact:latest”)
local_path = artifact.download(“./tmp”)
```

ArtifactsはW&BのArtifactsセクションにあり、エイリアスを使用して参照できます。これらのエイリアスは自動的に生成され（「latest」、「v2」、「v3」など）、または手動でログする際に作成されます（「best_accuracy」など）。

`wandb` run（`wandb.init` を通じて）を作成せずにアーティファクトをダウンロードするには、例えば分散環境や単純な推論において、代わりに[wandb API](https://docs.wandb.ai/ref/python/public-api) を使用してアーティファクトを参照します：

```python
artifact = wandb.Api().artifact(“user/project/artifact:latest”)
