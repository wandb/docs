---
title: TensorBoard
menu:
  default:
    identifier: tensorboard
    parent: integrations
weight: 430
---

{{< cta-button colabLink="https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard_and_Weights_and_Biases.ipynb" >}}

{{% alert %}}
W&B はマルチテナント SaaS 向けに埋め込み型 TensorBoard をサポートしています。
{{% /alert %}}

TensorBoard のログをクラウドにアップロードし、同僚やクラスメートと結果を素早く共有、分析内容を 1 か所にまとめて管理できます。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="TensorBoard インテグレーションのコード" >}}

## はじめる

```python
import wandb

# `sync_tensorboard=True` で wandb run を開始
wandb.init(project="my-project", sync_tensorboard=True) as run:
  # TensorBoard を使ったトレーニングコード
  ...

```

[TensorBoard インテグレーションの実例 run](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard) をご覧ください。

run が終了すると、W&B で TensorBoard のイベントファイルにアクセスでき、メトリクスをネイティブの W&B チャートで可視化できます。さらに、システムの CPU や GPU 利用率、`git` の状態、run で使用したターミナルコマンドなど、追加で役立つ情報も表示されます。

{{% alert %}}
W&B はすべての TensorFlow バージョンで TensorBoard をサポートしています。さらに PyTorch および TensorBoardX では TensorBoard 1.14 以降もサポートしています。
{{% /alert %}}

## よくある質問

### TensorBoard にログされていないメトリクスも W&B に保存できますか？

TensorBoard で記録されていないカスタムメトリクスを追加で保存したい場合は、コード内で `wandb.Run.log()` を呼び出します。`run.log({"custom": 0.8})` のように使います。

`synchronize Tensorboard`（Tensorboard 同期）時には `run.log()` の step 引数の設定は無効化されます。異なる step 数でメトリクスを記録したい場合は、以下のように step メトリクスとともに記録します：

`run.log({"custom": 0.8, "global_step": global_step})`

### `wandb` と併用する際に TensorBoard をどのように設定すればよいですか？

TensorBoard のパッチ方法をより細かく制御したい場合は、`wandb.init` に `sync_tensorboard=True` を渡す代わりに、`wandb.tensorboard.patch` を呼び出します。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
run = wandb.init()

# ノートブックで実行している場合、run を finish して tensorboard ログを W&B にアップロード
run.finish()
```

このメソッドに `tensorboard_x=False` を渡すことで、バニラの TensorBoard のみがパッチされるようにできます。PyTorch かつ TensorBoard >1.14 を利用している場合は `pytorch=True` を渡すことでパッチされます。これらのオプションには、各ライブラリのバージョンに応じたスマートなデフォルト設定があります。

デフォルトでは `tfevents` ファイルや `.pbtxt` ファイルも同期されます。これにより TensorBoard インスタンスを自動で起動できます。run ページで [TensorBoard タブ](https://www.wandb.com/articles/hosted-tensorboard) が表示されます。この挙動は `wandb.tensorboard.patch` に `save=False` を渡すことで無効にできます。

```python
import wandb

run = wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# ノートブック実行時は run を finish して tensorboard ログを W&B にアップロード
run.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer` を呼び出したり `torch.utils.tensorboard` で `SummaryWriter` を作成する**前に**、必ず `wandb.init()` または `wandb.tensorboard.patch` を呼び出してください。
{{% /alert %}}

### 過去の TensorBoard runs を同期するには？

既存の `tfevents` ファイルがローカルに保存されていて、それらを W&B にインポートしたい場合は、`wandb sync log_dir` を実行します。`log_dir` は `tfevents` ファイルを含むローカルディレクトリーです。

### Google Colab や Jupyter で TensorBoard を使うには？

Jupyter または Colab ノートブックでコードを実行する場合は、トレーニングの最後に `wandb.Run.finish()` を必ず呼び出してください。これにより run を終了し、tensorboard ログを W&B にアップロードできるようになります。`.py` スクリプト実行の場合は自動で終了するため、この作業は不要です。

ノートブック環境でシェルコマンドを実行する際は、`!wandb sync directoryname` のように先頭に `!` を付けてください。

### PyTorch で TensorBoard を使うには？

PyTorch の TensorBoard インテグレーションを使う場合は、PyTorch Profiler の JSON ファイルを手動でアップロードする必要がある場合があります。

```python
with wandb.init(project="my-project", sync_tensorboard=True) as run:
    run.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```

### クラウドに保存された tfevents ファイルも同期できますか？

`wandb` バージョン 0.20.0 以上では、S3・GCS・Azure に保存された `tfevents` ファイルも同期できます。`wandb` は各クラウドプロバイダー用のデフォルト認証情報を利用します。下記のコマンドに対応しています：

| クラウドプロバイダー | 認証情報                                   | ログディレクトリ形式                     |
| -------------------- | ---------------------------------------- | --------------------------------------- |
| S3                   | `aws configure`                         | `s3://bucket/path/to/logs`             |
| GCS                  | `gcloud auth application-default login` | `gs://bucket/path/to/logs`             |
| Azure                | `az login`[^1]                          | `az://account/container/path/to/logs`   |

[^1]: `AZURE_STORAGE_ACCOUNT` および `AZURE_STORAGE_KEY` の環境変数も設定する必要があります。