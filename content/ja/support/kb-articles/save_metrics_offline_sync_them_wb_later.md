---
title: Is it possible to save metrics offline and sync them to W&B later?
menu:
  support:
    identifier: ja-support-kb-articles-save_metrics_offline_sync_them_wb_later
support:
- experiments
- environment variables
- metrics
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、`wandb.init` は、メトリクス をリアルタイムで クラウド に同期する プロセス を開始します。オフラインで使用する場合は、2 つの 環境 変数 を設定して、オフラインモードを有効にし、後で同期します。

以下の 環境 変数 を設定します。

1. `WANDB_API_KEY=$KEY` 。`$KEY` は、[設定 ページ](https://app.wandb.ai/settings) の APIキー です。
2. `WANDB_MODE="offline"`

スクリプト でこれを実装する例を以下に示します。

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

作業が完了したら、次の コマンド を実行して、 データ を クラウド に同期します。

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="" >}}
