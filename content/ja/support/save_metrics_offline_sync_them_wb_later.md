---
title: Is it possible to save metrics offline and sync them to W&B later?
menu:
  support:
    identifier: ja-support-save_metrics_offline_sync_them_wb_later
tags:
- experiments
- environment variables
- metrics
toc_hide: true
type: docs
---

`wandb.init`はデフォルトでメトリクスをリアルタイムでクラウドに同期するプロセスを開始します。オフラインで使用する場合は、2つの環境変数を設定してオフラインモードを有効にし、後で同期させます。

以下の環境変数を設定します：

1. `WANDB_API_KEY=$KEY`、ここで`$KEY`はあなたの [settings page](https://app.wandb.ai/settings) からの APIキー です。
2. `WANDB_MODE="offline"`。

スクリプトでの実装例を示します：

```python
import wandb
import os

# 環境変数の設定
os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

# 設定
config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

# プロジェクトの初期化
wandb.init(project="offline-demo")

for i in range(100):
    # メトリクスのログ
    wandb.log({"accuracy": i})
```

以下にサンプルの端末出力を示します：

{{< img src="/images/experiments/sample_terminal_output.png" alt="" >}}

作業完了後、次のコマンドを実行してデータをクラウドに同期します：

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="" >}}