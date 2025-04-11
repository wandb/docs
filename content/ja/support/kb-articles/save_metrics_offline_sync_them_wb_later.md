---
title: メトリクスをオフラインで保存し、後で W&B に同期することはできますか？
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

`wandb.init` はデフォルトでメトリクスをリアルタイムでクラウドに同期するプロセスを開始します。オフラインで使用する場合は、オフラインモードを有効にし、後で同期できるように2つの環境変数を設定してください。

次の環境変数を設定します:

1. `WANDB_API_KEY=$KEY`、ここで `$KEY` はあなたの [settings page](https://app.wandb.ai/settings) から取得した APIキーです。
2. `WANDB_MODE="offline"`。

スクリプトでこれを実装する例を以下に示します:

```python
import wandb
import os

os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"  # あなたのAPIキーをここに
os.environ["WANDB_MODE"] = "offline"  # オフラインモードを設定

config = {
    "dataset": "CIFAR10",  # データセットを指定
    "machine": "offline cluster",  # オフラインクラスターを指定
    "model": "CNN",  # モデルを指定
    "learning_rate": 0.01,  # 学習率を指定
    "batch_size": 128,  # バッチサイズを指定
}

wandb.init(project="offline-demo")  # W&Bプロジェクトを初期化

for i in range(100):
    wandb.log({"accuracy": i})  # メトリクスをログ
```

サンプルのターミナル出力は以下の通りです:

{{< img src="/images/experiments/sample_terminal_output.png" alt="" >}}

作業が完了した後、データをクラウドに同期するために以下のコマンドを実行します:

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="" >}}