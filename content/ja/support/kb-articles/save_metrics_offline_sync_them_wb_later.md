---
title: メトリクスをオフラインで保存して、後で W&B に同期することはできますか？
menu:
  support:
    identifier: ja-support-kb-articles-save_metrics_offline_sync_them_wb_later
support:
- 実験管理
- 環境変数
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

デフォルトでは、`wandb.init` は メトリクス をリアルタイムに クラウド へ同期する プロセス を開始します。オフラインで使う場合は、オフライン モード を有効にして後で同期できるよう、2 つの 環境変数 を設定します。

次の 環境変数 を設定してください:

1. `WANDB_API_KEY=$KEY`。ここで、`$KEY` は [設定ページ](https://app.wandb.ai/settings) の APIキー です。
2. `WANDB_MODE="offline"`。

これを スクリプト で実装する例は次のとおりです:

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

with wandb.init(project="offline-demo") as run:
    for i in range(100):
        run.log({"accuracy": i})
```

以下はサンプルの ターミナル 出力です:

{{< img src="/images/experiments/sample_terminal_output.png" alt="オフライン モードのターミナル出力" >}}

作業が完了したら、次の コマンド を実行して データ を クラウド に同期します:

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="クラウド 同期のターミナル出力" >}}