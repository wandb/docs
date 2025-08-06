---
title: メトリクスをオフラインで保存して、後で W&B に同期することは可能ですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
- 環境変数
- メトリクス
---

デフォルトでは、`wandb.init` はメトリクスをリアルタイムでクラウドに同期するプロセスを開始します。オフラインで使用する場合は、2つの環境変数を設定してオフラインモードを有効にし、後で同期できます。

以下の環境変数を設定してください：

1. `WANDB_API_KEY=$KEY`（ここで `$KEY` は [settings page](https://app.wandb.ai/settings) から取得した APIキー です）
2. `WANDB_MODE="offline"`

この設定をスクリプトで実装する例は以下の通りです。

```python
import wandb
import os

# APIキーとオフラインモードを環境変数で指定
os.environ["WANDB_API_KEY"] = "YOUR_KEY_HERE"
os.environ["WANDB_MODE"] = "offline"

config = {
    "dataset": "CIFAR10",
    "machine": "offline cluster",
    "model": "CNN",
    "learning_rate": 0.01,
    "batch_size": 128,
}

# offline-demo というプロジェクトで run を開始
with wandb.init(project="offline-demo") as run:
    for i in range(100):
        # 精度 (accuracy) をログ
        run.log({"accuracy": i})
```

下記はサンプルのターミナル出力です。

{{< img src="/images/experiments/sample_terminal_output.png" alt="オフラインモードのターミナル出力" >}}

作業が完了したら、次のコマンドを実行してデータをクラウドに同期します。

```shell
wandb sync wandb/dryrun-folder-name
```

{{< img src="/images/experiments/sample_terminal_output_cloud.png" alt="クラウド同期時のターミナル出力" >}}