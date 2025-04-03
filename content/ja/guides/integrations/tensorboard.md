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
W&B は、W&B マルチテナント SaaS 用の埋め込み TensorBoard をサポートしています。
{{% /alert %}}

TensorBoard の ログ を クラウド にアップロードし、同僚やクラスメートとすばやく 結果 を共有して、 分析 を一元的な場所に保管できます。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="" >}}

## はじめに

```python
import wandb

# `sync_tensorboard=True` で wandb の run を開始します。
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard を使用したトレーニングコード
...

# [オプション] W&B に TensorBoard の ログ をアップロードするために、wandb の run を終了します ( Colabノートブック で実行している場合)。
wandb.finish()
```

[例](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard) を確認してください。

run が完了すると、W&B で TensorBoard イベントファイルに アクセス し、ネイティブの W&B チャートで メトリクス を視覚化できます。さらに、システムの CPU や GPU の使用率、`git` の状態、run が使用した ターミナル コマンドなど、役立つ追加情報も表示できます。

{{% alert %}}
W&B は、すべての TensorFlow バージョンで TensorBoard をサポートしています。W&B は、PyTorch および TensorBoardX を使用した TensorBoard 1.14 以降もサポートしています。
{{% /alert %}}

## よくある質問

### TensorBoard に ログ 記録されない メトリクス を W&B に ログ 記録するにはどうすればよいですか？

TensorBoard に ログ 記録されていない追加のカスタム メトリクス を ログ 記録する必要がある場合は、コードで `wandb.log` を呼び出すことができます。`wandb.log({"custom": 0.8})`

Tensorboard を同期すると、`wandb.log` の step 引数 の設定はオフになります。別の step カウントを設定する場合は、step メトリクス とともに メトリクス を ログ 記録できます。

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb` で TensorBoard を使用している場合、TensorBoard をどのように構成しますか？

TensorBoard の パッチ 適用方法をより詳細に制御する場合は、`wandb.init` に `sync_tensorboard=True` を渡す代わりに、`wandb.tensorboard.patch` を呼び出すことができます。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# W&B に TensorBoard の ログ をアップロードするために、wandb の run を終了します ( Colabノートブック で実行している場合)。
wandb.finish()
```

TensorBoard > 1.14 を PyTorch で使用している場合は、バニラ TensorBoard に パッチ が適用されていることを確認するために `tensorboard_x=False` をこの メソッド に渡し、 パッチ が適用されていることを確認するために `pytorch=True` を渡すことができます。これらのオプションには両方とも、これらのライブラリのどの バージョン がインポートされたかに応じて、スマートなデフォルトがあります。

デフォルトでは、`tfevents` ファイルと `.pbtxt` ファイルも同期します。これにより、お客様に代わって TensorBoard インスタンスを ローンンチ することができます。run ページに [TensorBoard タブ](https://www.wandb.com/articles/hosted-tensorboard) が表示されます。この 振る舞い は、`save=False` を `wandb.tensorboard.patch` に渡すことでオフにできます。

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# Colabノートブック で実行している場合は、wandb の run を終了して、TensorBoard の ログ を W&B にアップロードします。
wandb.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer` を呼び出すか、`torch.utils.tensorboard` を介して `SummaryWriter` を構築する **前に** 、`wandb.init` または `wandb.tensorboard.patch` のいずれかを呼び出す必要があります。
{{% /alert %}}

### 履歴 TensorBoard の run を同期するにはどうすればよいですか？

既存の `tfevents` ファイルがローカルに保存されていて、それらを W&B にインポートする場合は、`wandb sync log_dir` を実行します。ここで、`log_dir` は `tfevents` ファイルを含むローカル ディレクトリー です。

### Google Colab または Jupyter を TensorBoard で使用するにはどうすればよいですか？

Jupyter または Colabノートブック でコードを実行している場合は、トレーニングの最後に `wandb.finish()` を呼び出すようにしてください。これにより、wandb の run が終了し、TensorBoard の ログ が W&B にアップロードされて視覚化できるようになります。wandb は スクリプト の完了時に自動的に完了するため、`.py` スクリプト を実行する場合はこれは必要ありません。

ノートブック 環境で シェル コマンド を実行するには、`!wandb sync directoryname` のように `!` を先頭に付ける必要があります。

### PyTorch を TensorBoard で使用するにはどうすればよいですか？

PyTorch の TensorBoard インテグレーション を使用する場合は、PyTorch Profiler JSON ファイルを手動でアップロードする必要がある場合があります。

```python
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```
