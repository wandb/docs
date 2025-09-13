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
W&B は W&B マルチテナント SaaS で埋め込み TensorBoard をサポートしています。
{{% /alert %}}

TensorBoard のログを クラウド にアップロードし、結果 を同僚やクラスメイトとすばやく共有して、分析 を 1 箇所に集約しましょう。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="TensorBoard インテグレーションのコード" >}}

## はじめに

```python
import wandb

# `sync_tensorboard=True` で wandb の run を開始します
wandb.init(project="my-project", sync_tensorboard=True) as run:
  # TensorBoard を使った トレーニング コード
  ...

```

[TensorBoard インテグレーションの run の例](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard) を確認してください。

run が完了すると、W&B で TensorBoard のイベントファイルに アクセス でき、ネイティブな W&B チャートで メトリクス を可視化できます。さらに、システムの CPU や GPU 使用率、`git` の状態、run が使った ターミナル コマンド などの有用な 情報 もあわせて表示されます。

{{% alert %}}
W&B はすべての バージョン の TensorFlow で TensorBoard をサポートします。W&B は PyTorch で TensorBoard 1.14 以降、および TensorBoardX もサポートしています。
{{% /alert %}}

## よくある質問

### TensorBoard に ログ されていない メトリクス を W&B に ログ するには？

TensorBoard に ログ されていないカスタム メトリクス を追加で ログ したい場合は、コード内で `wandb.Run.log()` を呼び出します。例: `run.log({"custom": 0.8})`

TensorBoard を同期しているときは、`run.log()` の step 引数の設定は無効になります。異なるステップ数を設定したい場合は、次のようにステップ メトリクス と一緒に ログ してください:

`run.log({"custom": 0.8, "global_step": global_step})`

### `wandb` と併用する際に TensorBoard を どのように 設定 すればよいですか？

TensorBoard のパッチ方法をより詳細に制御したい場合は、`wandb.init` に `sync_tensorboard=True` を渡す代わりに、`wandb.tensorboard.patch` を呼び出します。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
run = wandb.init()

# ノートブック で実行している場合は、TensorBoard のログを W&B にアップロードするために wandb の run を終了します
run.finish()
```

この メソッド に `tensorboard_x=False` を渡すと、バニラの TensorBoard がパッチされるようにできます。PyTorch とともに TensorBoard > 1.14 を使っている場合は、`pytorch=True` を渡して確実にパッチされるようにできます。これらのオプションには、インポートされているライブラリの バージョン に応じたスマートなデフォルトが用意されています。

デフォルトでは、`tfevents` ファイルと `.pbtxt` ファイルも同期します。これにより、こちらで TensorBoard インスタンスを起動できます。run ページに [TensorBoard タブ](https://www.wandb.com/articles/hosted-tensorboard) が表示されます。この 振る舞い は、`wandb.tensorboard.patch` に `save=False` を渡すことでオフにできます。

```python
import wandb

run = wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# ノートブック で実行している場合は、TensorBoard のログを W&B にアップロードするために wandb の run を終了します
run.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer` を呼び出す前、または `torch.utils.tensorboard` で `SummaryWriter` を構築する前に、`wandb.init()` か `wandb.tensorboard.patch` のいずれかを必ず呼び出してください。
{{% /alert %}}

### 過去の TensorBoard run を同期するには？

ローカルに保存した既存の `tfevents` ファイルを W&B にインポートしたい場合は、`wandb sync log_dir` を実行します。ここで `log_dir` は `tfevents` ファイルを含むローカル ディレクトリー です。

### Google Colab や Jupyter で TensorBoard を使うには？

Jupyter や Colab の ノートブック でコードを実行している場合は、トレーニング の最後に `wandb.Run.finish()` を必ず呼び出してください。これにより wandb の run が終了し、TensorBoard のログが W&B にアップロードされて可視化できるようになります。`.py` スクリプトを実行する場合は不要で、スクリプトの終了時に wandb は自動的に終了します。

ノートブック 環境でシェル コマンド を実行するには、先頭に `!` を付けます。例: `!wandb sync directoryname`

### PyTorch と TensorBoard を併用するには？

PyTorch の TensorBoard インテグレーションを使用している場合、PyTorch Profiler の JSON ファイルを手動でアップロードする必要がある場合があります。

```python
with wandb.init(project="my-project", sync_tensorboard=True) as run:
    run.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```

### クラウド に保存された tfevents ファイルを同期できますか？

`wandb` 0.20.0 以降は、S3、GCS、または Azure に保存された `tfevents` ファイルの同期をサポートしています。`wandb` は各 クラウド プロバイダーのデフォルト資格情報を使用します。対応するコマンドは次のとおりです:

| クラウド プロバイダー | 資格情報                                 | ログ ディレクトリー の形式            |
| -------------- | --------------------------------------- | ------------------------------------- |
| S3             | `aws configure`                         | `s3://bucket/path/to/logs`            |
| GCS            | `gcloud auth application-default login` | `gs://bucket/path/to/logs`            |
| Azure          | `az login`[^1]                          | `az://account/container/path/to/logs` |

[^1]: `AZURE_STORAGE_ACCOUNT` と `AZURE_STORAGE_KEY` の 環境変数 も設定する必要があります。