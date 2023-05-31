---
displayed_sidebar: ja
---
# TensorBoard

## 1行のコードでホストされたTensorBoard

Weight & Biasesを使えば、TensorBoardのログを簡単にクラウドにアップロードして、同僚やクラスメートとすばやく結果を共有し、分析を一元化した場所で管理することができます。

**このノートブックで今すぐ始めましょう:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard\_and\_Weights\_and\_Biases.ipynb)

![](/images/integrations/tensorboard_oneline_code.webp)

### コード1行を追加するだけ

```python
import wandb
# `sync_tensorboard=True`でwandb runを開始
wandb.init(project='my-project', sync_tensorboard=True)

# TensorBoardを使用したトレーニングコード
...

# [オプション]TensorboardのログをW&Bにアップロードするために、wandb runを終了（ノートブック内で実行の場合）
wandb.finish()
```

[**Weights & BiasesでホストされたTensorboardの例はこちらを参照**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

wandb runが終了すると、TensorBoardのイベントファイルがWeights & Biasesにアップロードされます。これらのメトリクスは、マシンのCPUやGPU利用率、gitの状態、使用されるターミナルコマンドなどの有用な情報とともに、Weights & Biasesのネイティブチャートに**ログされます**。

:::info
Weights & Biasesは、全バージョンのTensorFlowでTensorBoardに対応しています。また、W&BはTensorBoard > 1.14のPyTorchおよびTensorBoardXもサポートしています。
:::
## 一般的な質問

### TensorBoardにログされていないW&Bへのメトリクスの記録方法は？

TensorBoardにログされていない追加のカスタムメトリクスをログする必要がある場合は、コードで `wandb.log` を呼び出すことができます。`wandb.log({"custom": 0.8})`

TensorBoardと同期する際に、`wandb.log`の引数にstepを設定することはできません。異なるステップカウントを設定したい場合は、stepメトリクスとともにメトリクスをログできます。

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb` と一緒に TensorBoard を使用している場合、TensorBoard の設定方法は？

TensorBoardにパッチを当てる方法をより制御したい場合は、`wandb.init` に `sync_tensorboard=True` を渡す代わりに、`wandb.tensorboard.patch` を呼び出すことができます。

```python
import wandb
wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# ノートブックで実行している場合、wandbの実行を終了してtensorboardのログをW&Bにアップロードする
wandb.finish()
```

このメソッドに `tensorboard_x=False` を渡すことで、バニラのTensorBoardにパッチが当たることが確実になります。TensorBoard > 1.14 を PyTorch と一緒に使用している場合は、`pytorch=True` を渡すことでパッチが適用されることを確認できます。これらのオプションは、インポートされたライブラリのバージョンに応じて賢いデフォルトが設定されています。

デフォルトでは、`tfevents`ファイルと`.pbtxt`ファイルも同期されます。これにより、代わりにTensorBoardインスタンスを起動できます。実行ページで[TensorBoardタブ](https://www.wandb.com/articles/hosted-tensorboard)が表示されます。この振る舞いは、`wandb.tensorboard.patch` に `save=False` を渡すことで無効にできます。

```python
import wandb
wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)
# ノートブックで実行している場合は、wandb runを終了して、TensorBoardのログをW&Bにアップロード

wandb.finish()

```

:::caution

`tf.summary.create_file_writer`を呼び出すか、`torch.utils.tensorboard`で`SummaryWriter`を構築する**前**に、`wandb.init`か`wandb.tensorboard.patch`を呼び出す必要があります。

:::

### 以前のTensorBoardのランを同期する

すでにローカルに保存されている`tfevents`ファイルがある場合で、それらをW&Bにインポートしたい場合は、`wandb sync log_dir`を実行し、`log_dir` は `tfevents`ファイルが含まれるローカルディレクトリーにしてください。

### Google Colab, Jupyter、そして TensorBoard

JupyterやColabのノートブックでコードを実行している場合は、トレーニングの終了時に`wandb.finish()`を呼び出してください。これにより、wandbのrunを終了し、TensorBoardのログがW&Bにアップロードされて、可視化されます。これは、`.py`スクリプトを実行する際には自動的にwandbが終了するので、不要です。

ノートブック環境でシェルコマンドを実行するには、`!`を追加して、`!wandb sync directoryname` のようにしなければなりません。

### PyTorchとTensorBoard

もしPyTorchのTensorBoard統合を利用している場合、PyTorchプロファイラのJSONファイルを手動でアップロードする必要があるかもしれません。

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```