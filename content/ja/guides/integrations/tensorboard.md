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
W&B は、W&B Multi-tenant SaaS 用の埋め込み TensorBoard をサポートしています。
{{% /alert %}}

TensorBoard の ログ を クラウド にアップロードして、同僚やクラスメートとすばやく 結果 を共有し、1 つの集中管理された場所に 分析 を保持します。

{{< img src="/images/integrations/tensorboard_oneline_code.webp" alt="" >}}

## 始めましょう

```python
import wandb

# `sync_tensorboard=True` を指定して wandb run を開始します
wandb.init(project="my-project", sync_tensorboard=True)

# TensorBoard を使用した トレーニング コード
...

# [オプション] TensorBoard の ログ を W&B にアップロードするには、wandb run を終了します ( Notebook で 実行 している 場合 )
wandb.finish()
```

[例](https://wandb.ai/rymc/simple-tensorboard-example/runs/oab614zf/tensorboard)をご覧ください。

run が完了すると、W&B で TensorBoard イベントファイルに アクセス できます。また、システムの CPU や GPU の 使用率、`git` の 状態、run で使用された ターミナル コマンド など、追加の 役立つ 情報 とともに、ネイティブの W&B チャートで メトリクス を視覚化できます。

{{% alert %}}
W&B は、すべての バージョン の TensorFlow で TensorBoard をサポートしています。W&B は、PyTorch および TensorBoardX を使用した TensorBoard 1.14 以降もサポートしています。
{{% /alert %}}

## よくある質問

### TensorBoard に ログ されない メトリクス を W&B に ログ するにはどうすればよいですか?

TensorBoard に ログ されていない追加の カスタム メトリクス を ログ する必要がある 場合は、コード で `wandb.log` を呼び出すことができます `wandb.log({"custom": 0.8})`

Tensorboard を 同期 すると、`wandb.log` の step 引数 の 設定 は オフ になります。異なる ステップ 数 を 設定 したい 場合は、次のようにステップ メトリクス を使用して メトリクス を ログ できます。

`wandb.log({"custom": 0.8, "global_step": global_step})`

### `wandb` と一緒に使用している場合、Tensorboard をどのように構成しますか?

TensorBoard の パッチ 適用方法をより詳細に制御したい 場合は、`wandb.init` に `sync_tensorboard=True` を渡す代わりに、`wandb.tensorboard.patch` を呼び出すことができます。

```python
import wandb

wandb.tensorboard.patch(root_logdir="<logging_directory>")
wandb.init()

# TensorBoard の ログ を W&B にアップロードするには、wandb run を終了します ( Notebook で 実行 している 場合 )
wandb.finish()
```

TensorBoard > 1.14 を PyTorch で使用している 場合は、`tensorboard_x=False` をこの メソッド に渡して、バニラ TensorBoard に パッチ が 適用 されていることを確認できます。`pytorch=True` を渡して、パッチ が 適用 されていることを確認できます。これらの オプション は 両方 とも、これらの ライブラリ の バージョン に応じて スマート な デフォルト を持ちます。

デフォルト では、`tfevents` ファイルとすべての `.pbtxt` ファイルも 同期 します。これにより、ユーザー に代わって TensorBoard インスタンス を 起動 できます。run ページに [TensorBoard タブ](https://www.wandb.com/articles/hosted-tensorboard) が 表示 されます。この 振る舞い は、`save=False` を `wandb.tensorboard.patch` に渡すことで オフ にできます。

```python
import wandb

wandb.init()
wandb.tensorboard.patch(save=False, tensorboard_x=True)

# notebook で 実行 している 場合は、wandb run を終了して tensorboard ログ を W&B にアップロードします
wandb.finish()
```

{{% alert color="secondary" %}}
`tf.summary.create_file_writer` を呼び出すか、`torch.utils.tensorboard` 経由 で `SummaryWriter` を構築する **前に** 、`wandb.init` または `wandb.tensorboard.patch` のいずれかを呼び出す必要があります。
{{% /alert %}}

### 履歴 TensorBoard run を 同期 するにはどうすればよいですか?

既存の `tfevents` ファイルがローカルに 保存 されていて、W&B に インポート したい 場合は、`wandb sync log_dir` を 実行 できます。`log_dir` は `tfevents` ファイルを 含む ローカル ディレクトリー です。

### Google Colab または Jupyter を TensorBoard と一緒に使用するにはどうすればよいですか?

Jupyter または Colab ノートブック で コード を 実行 している 場合は、 トレーニング の 最後に 必ず `wandb.finish()` を呼び出してください。これにより、wandb run が終了し、TensorBoard ログ が W&B にアップロードされ、視覚化できるようになります。これは、`.py` スクリプト を 実行 する 場合 、スクリプト の 終了 時に wandb が自動的に終了するため、必要ありません。

notebook 環境 で シェル コマンド を 実行 するには、`!wandb sync directoryname` のように、`!` を先頭に付ける必要があります。

### PyTorch を TensorBoard と一緒に使用するにはどうすればよいですか?

PyTorch の TensorBoard インテグレーション を使用する 場合 は、PyTorch Profiler JSON ファイルを 手動 でアップロードする 必要がある 場合 があります。

```python
wandb.save(glob.glob(f"runs/*.pt.trace.json")[0], base_path=f"runs")
```
