---
title: TensorBoard
menu:
  default:
    identifier: ja-guides-integrations-tensorboard
    parent: integrations
weight: 430
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard_and_Weights_and_Biases.ipynb" >}}

{{% alert %}}
W&B は、マルチテナント SaaS 環境で TensorBoard の埋め込みをサポートしています。
{{% /alert %}}

TensorBoard のログをクラウドにアップロードし、同僚やクラスメートとすばやく結果を共有し、分析を 1 つの場所にまとめておくことができます。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="TensorBoard インテグレーションコード" >}}

## はじめに

```python
import wandb

# `sync_tensorboard=True` で wandb run を開始する
wandb.init(project="my-project", sync_tensorboard=True) as run:
  # TensorBoard を使ったトレーニングコード
  ...

```

[TensorBoard インテグレーションのサンプル run](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard) を確認してください。

run が終了すると、W&B 上で TensorBoard のイベントファイルにア クセスできるようになり、ネイティブな W&B チャートでメトリクスを可視化できます。他にも、システムの CPU や GPU 使用率、`git` の状態、run で使用したターミナルコマンドなども併せて確認できます。

{{% alert %}}
W&B は、すべての TensorFlow バージョンで TensorBoard をサポートしています。また、PyTorch および TensorBoardX では TensorBoard 1.14 以上に対応しています。
{{% /alert %}}

## よくある質問

### TensorBoard に記録されていないメトリクスを W&B にログするには？

TensorBoard に記録されていないカスタムメトリクスを追加でログしたい場合は、コード内で `wandb.Run.log()` を呼び出します。  
`run.log({"custom": 0.8})`

TensorBoard 同期中は `run.log()` で step 引数をセットする機能は無効です。もし異なる step カウントを指定したい場合、次のように step メトリクスも併せてログできます。

`run.log({"custom": 0.8, "global_step": global_step})`

### `wandb` と一緒に TensorBoard を使う場合の設定方法を教えてください

TensorBoard のパッチ適用方法をより細かく指定したい場合、`wandb.init` の `sync_tensorboard=True` の代わりに `wandb.tensorboard.patch` を呼び出すことができます。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
run = wandb.init()

# ノートブック環境で実行している場合は run を終了して TensorBoard のログをアップロード
run.finish()
```

vanilla の TensorBoard のみをパッチしたい場合は、このメソッドに `tensorboard_x=False` を渡してください。PyTorch 1.14 より新しいバージョンの TensorBoard を使い PyTorch と組み合わせる場合は、`pytorch=True` を指定することで適切なパッチが適用されます。これらのオプションは、読み込まれている各ライブラリのバージョンによってスマートにデフォルトが変わります。

デフォルトでは `tfevents` ファイルと `.pbtxt` ファイルも同期対象です。これにより TensorBoard インスタンスを W&B が自動でローンチします。run ページには [TensorBoard タブ](https://www.wandb.com/articles/hosted-tensorboard) が表示されます。この振る舞いは `wandb.tensorboard.patch` に `save=False` を渡すことでオフにできます。

```python
import wandb

run = wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# ノートブック環境で実行している場合、run を終了して TensorBoard のログをアップロード
run.finish()
```

{{% alert color="secondary" %}}
必ず `tf.summary.create_file_writer` を呼ぶ前、または `torch.utils.tensorboard` で `SummaryWriter` を作成する前に `wandb.init()` または `wandb.tensorboard.patch` を呼んでください。
{{% /alert %}}

### 過去の TensorBoard run を同期するには？

すでにローカルに保存してある `tfevents` ファイルを W&B にインポートしたい場合は、`wandb sync log_dir` を実行してください。`log_dir` には `tfevents` ファイルが含まれるローカルディレクトリーのパスを指定します。

### Google Colab や Jupyter で TensorBoard を使うには？

Jupyter や Colabノートブック環境でコードを実行する場合は、トレーニング終了時に `wandb.Run.finish()` を呼ぶようにしてください。これにより wandb run が終了し、TensorBoard のログが W&B にアップロードされて可視化できるようになります。`.py` スクリプトの場合、スクリプト終了時に wandb が自動で終了処理をするため明示的な finish は不要です。

ノートブック環境でシェルコマンドを実行するには、先頭に `!` を付けてください (例: `!wandb sync directoryname`)。

### PyTorch で TensorBoard を利用するには？

PyTorch の TensorBoard インテグレーションを使用する場合、PyTorch Profiler の JSON ファイルは手動でアップロードする必要があるかもしれません。

```python
with wandb.init(project="my-project", sync_tensorboard=True) as run:
    run.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```

### クラウドに保存した tfevents ファイルも同期できますか？

`wandb` 0.20.0 以降では、S3・GCS・Azure に保存されている `tfevents` ファイルの同期がサポートされています。それぞれのクラウドプロバイダーに対応したデフォルトの認証情報が使用され、対応するコマンドは次のとおりです。

| クラウドプロバイダー | 認証コマンド                                     | ログディレクトリーの形式                |
| ------------------- | ---------------------------------------------- | --------------------------------------- |
| S3                  | `aws configure`                               | `s3://bucket/path/to/logs`              |
| GCS                 | `gcloud auth application-default login`        | `gs://bucket/path/to/logs`              |
| Azure               | `az login`[^1]                                | `az://account/container/path/to/logs`   |

[^1]: `AZURE_STORAGE_ACCOUNT` と `AZURE_STORAGE_KEY` の環境変数も設定する必要があります。