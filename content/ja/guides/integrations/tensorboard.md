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
W&Bは、W&BマルチテナントSaaSで埋め込みTensorBoardをサポートしています。
{{% /alert %}}

TensorBoardログをクラウドにアップロードして、同僚やクラスメイトと素早く結果を共有し、一元化された場所で分析を保持できます。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="" >}}

## 始めましょう

```python
import wandb

# `sync_tensorboard=True` を使ってwandb runを開始します
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoardを使用したトレーニングコード
...

# [オプション] ノートブックで実行している場合は、wandb runを終了して、TensorBoardログをW&Bにアップロードします
wandb.finish()
```

[例](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)を確認してください。

runが終了すると、W&BでTensorBoardイベントファイルにアクセスでき、システムのCPUやGPUの利用状況、`git`の状態、runが使用したターミナルコマンドなどの便利な情報とともに、ネイティブなW&Bチャートでメトリクスを視覚化できます。

{{% alert %}}
W&Bは、すべてのバージョンのTensorFlowでTensorBoardをサポートしています。また、W&Bは、PyTorchとTensorBoardXでTensorBoard 1.14以降をサポートしています。
{{% /alert %}}

## よくある質問

### TensorBoardにログされていないメトリクスをW&Bにログするにはどうすればいいですか？

TensorBoardにログされていないカスタムメトリクスを追加でログする必要がある場合、コード内で`wandb.log`を呼び出すことができます `wandb.log({"custom": 0.8})`

TensorBoardと同期するとき、`wandb.log`でのstep引数の設定は無効になります。異なるステップ数を設定したい場合は、メトリクスをステップメトリクスと一緒にログできます：

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`と一緒に使用する際、TensorBoardをどのように設定しますか？

TensorBoardのパッチを制御したい場合は、`wandb.init`に`sync_tensorboard=True`を渡す代わりに`wandb.tensorboard.patch`を呼び出すことができます。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# ノートブックで実行している場合は、wandb runを終了してTensorBoardのログをW&Bにアップロードします
wandb.finish()
```

このメソッドに`tensorboard_x=False`を渡すことでプレーンなTensorBoardがパッチされることを確認できます。TensorBoard > 1.14をPyTorchで使用している場合は、確実にパッチされるように`pytorch=True`を渡すことができます。これらのオプションは、インポートされたこれらのライブラリーのバージョンに応じて賢いデフォルトを持っています。

デフォルトでは、`tfevents`ファイルと任意の`.pbtxt`ファイルも同期しています。これにより、あなたに代わってTensorBoardインスタンスを起動することができます。runページで[TensorBoardタブ](https://www.wandb.com/articles/hosted-tensorboard)を見ることができます。この振る舞いは、`wandb.tensorboard.patch`に`save=False`を渡すことでオフにできます。

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# ノートブックで実行している場合は、wandb runを終了してTensorBoardのログをW&Bにアップロードします
wandb.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer`を呼び出す前または`torch.utils.tensorboard`経由で`SummaryWriter`を構築する前に、`wandb.init`または`wandb.tensorboard.patch`のいずれかを呼び出す必要があります。
{{% /alert %}}

### 過去のTensorBoard runをどのように同期しますか？

ローカルに保存されている既存の`tfevents`ファイルがあり、それをW&Bにインポートしたい場合は、`wandb sync log_dir`を実行できます。ここで`log_dir`は`tfevents`ファイルを含むローカルディレクトリです。

### TensorBoardを使用するためにGoogle ColabまたはJupyterをどのように使用しますか？

JupyterまたはColabノートブックでコードを実行する場合、トレーニング終了時に`wandb.finish()`を必ず呼び出してください。これにより、wandb runが終了し、TensorBoardのログがW&Bにアップロードされ、可視化できるようになります。スクリプトが終了するとwandbが自動的に終了するため、`.py`スクリプトを実行している場合は必要ありません。

ノートブック環境でシェルコマンドを実行するには、`!`を前置する必要があります。例：`!wandb sync directoryname`

### PyTorchをTensorBoardでどのように使用しますか？

PyTorchのTensorBoardインテグレーションを使用する場合は、PyTorch Profiler JSONファイルを手動でアップロードする必要があるかもしれません。

```python
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```