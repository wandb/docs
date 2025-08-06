---
title: メトリクスをオフラインで保存し、後から W&B に同期することはできますか？
menu:
  support:
    identifier: ja-support-kb-articles-save_metrics_offline_sync_them_wb_later
support:
- 実験
- 環境変数
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、`wandb.init` はメトリクスをリアルタイムでクラウドに同期するプロセスを開始します。オフラインで利用したい場合は、2つの環境変数を設定してオフラインモードを有効にし、後で同期できます。

以下の環境変数を設定してください：

1. `WANDB_API_KEY=$KEY`（ここで `$KEY` は [settings page](https://app.wandb.ai/settings) から取得した APIキー です）
2. `WANDB_MODE="offline"`

この設定をスクリプト内で実装する例を示します：

```python
import wandb
import os

# 環境変数に APIキー とオフラインモードを設定
os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

with wandb.init(project="offline-demo") as run:
    for i in range(100):
        run.log({"accuracy": i})  # メトリクスをログ
```

サンプルのターミナル出力例は以下の通りです：

{{< img src="/images/experiments/sample_terminal_output.png" alt="Offline mode terminal output" >}}

作業が完了したら、次のコマンドを実行してデータをクラウドに同期します：

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="Cloud sync terminal output" >}}