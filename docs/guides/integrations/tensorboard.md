---
displayed_sidebar: default
---


# TensorBoard

## コード行1つでホストされたTensorBoard

Weights & Biasesを使用すると、TensorBoardログをクラウドに簡単にアップロードでき、同僚やクラスメートと結果をすぐに共有し、分析を1つの中央の場所にまとめておくことができます。

**このノートブックで今すぐ始めましょう:** [**Colabノートブックで試す →**](https://github.com/wandb/examples/blob/master/colabs/tensorboard/TensorBoard\_and\_Weights\_and\_Biases.ipynb)

![](/images/integrations/tensorboard_oneline_code.webp)

### コード行1つを追加するだけ

```python
import wandb

# `sync_tensorboard=True`でwandb runを開始
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoardを使用したトレーニングコード
...

# [オプション]wandb runを終了してtensorboardのログをW&Bにアップロード（ノートブックで実行している場合）
wandb.finish()
```

[**Weights & BiasesでホストされるTensorboardの例はこちら**](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)

wandb runが終了すると、TensorBoardのイベントファイルがWeights & Biasesにアップロードされます。これらのメトリクスは、ネイティブのWeights & Biasesチャートにもログされ、マシンのCPUやGPUの使用状況、git状態、使用されたターミナルコマンドなどの有用な情報も一緒に記録されます。

:::info
Weights & BiasesはすべてのバージョンのTensorFlowでTensorBoardをサポートしています。W&BはPyTorchのTensorBoard > 1.14およびTensorBoardXもサポートしています。
:::

## よくある質問

### TensorBoardにログされていないメトリクスをW&Bにログするにはどうすればよいですか？

TensorBoardにログされていないカスタムメトリクスをログする必要がある場合、コード内で`wandb.log`を呼び出すことができます `wandb.log({"custom": 0.8})`

Tensorboardと同期している場合、`wandb.log`でステップ引数を設定することはできません。異なるステップカウントを設定したい場合は、以下のようにステップメトリクスと一緒にログを記録することができます。

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb`を使用しているときにTensorboardをどのように設定しますか？

TensorBoardのパッチ適用方法をより詳細に制御したい場合、`wandb.init`に`sync_tensorboard=True`を渡す代わりに `wandb.tensorboard.patch`を呼び出すことができます。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# ノートブックで実行している場合、wandb runを終了してtensorboardのログをW&Bにアップロード
wandb.finish()
```

`tensorboard_x=False`をこのメソッドに渡すと、標準のTensorBoardがパッチされます。TensorBoard > 1.14でPyTorchを使用している場合、`pytorch=True`を渡すとパッチが確実に適用されます。これらのオプションには、インポートされたライブラリのバージョンに応じたスマートなデフォルト設定があります。

デフォルトでは、`tfevents`ファイルとすべての`.pbtxt`ファイルも同期します。これにより、あなたの代わりにTensorBoardインスタンスを起動できます。runページに[TensorBoardタブ](https://www.wandb.com/articles/hosted-tensorboard)が表示されます。この振る舞いは`save=False`を`wandb.tensorboard.patch`に渡すことで無効にできます。

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# ノートブックで実行している場合、wandb runを終了してtensorboardのログをW&Bにアップロード
wandb.finish()
```

:::caution
`tf.summary.create_file_writer`を呼び出す前、または`torch.utils.tensorboard`を使用して`SummaryWriter`を構築する前に、`wandb.init`または`wandb.tensorboard.patch`を呼び出す必要があります。
:::

### 以前のTensorBoard Runsの同期

ローカルに保存されている既存の`tfevents`ファイルをW&Bにインポートしたい場合、`wandb sync log_dir`を実行できます。ここで`log_dir`は`tfevents`ファイルを含むローカルディレクトリです。

### Google Colab, JupyterとTensorBoard

JupyterまたはColabノートブックでコードを実行している場合、トレーニング終了時に必ず`wandb.finish()`を呼び出してください。これにより、wandb runが終了し、TensorBoardのログがW&Bにアップロードされ、可視化できるようになります。 `.py`スクリプトを実行している場合は、スクリプトが終了すると自動的にwandbが終了するため、これは必要ありません。

ノートブック環境でシェルコマンドを実行するには、`!`を前に付ける必要があります。例: `!wandb sync directoryname`.

### PyTorchとTensorBoard

PyTorchのTensorBoardインテグレーションを使用する場合、PyTorch Profiler JSONファイルを手動でアップロードする必要がある場合があります**:**

```
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```