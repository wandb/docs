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

デフォルトでは、`wandb.init` は、メトリクスをリアルタイムで クラウド に同期する プロセス を開始します。オフラインで使用するには、2 つの 環境 変数を設定して、オフライン モードを有効にし、後で同期します。

次の 環境 変数を設定します。

1. `WANDB_API_KEY=$KEY` 。 `$KEY` は、[設定 ページ](https://app.wandb.ai/settings) の APIキー です。
2. `WANDB_MODE="offline"` 。

以下は、これを スクリプト で実装する例です。

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

wandb.init(project="offline-demo")

for i in range(100):
    wandb.log({"accuracy": i})
```

ターミナル の出力例を以下に示します。

{{< img src="/images/experiments/sample_terminal_output.png" alt="" >}}

作業完了後、次の コマンド を実行して、データを クラウド に同期します。

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="" >}}
