---
title: What does wandb.init do to my training process?
menu:
  support:
    identifier: ja-support-wandbinit_training_process
tags:
- environment variables
- experiments
toc_hide: true
type: docs
---

`wandb.init()` が トレーニングスクリプト 内で実行されると、API コールによりサーバー上に run オブジェクトが作成されます。新しいプロセスが開始され、メトリクスをストリーミングおよび収集し、主要なプロセスが通常どおり機能できるようになります。スクリプトはローカルファイルに書き込み、一方で別のプロセスがデータをサーバーにストリーミングします。これにはシステムメトリクスも含まれます。ストリーミングをオフにするには、トレーニングディレクトリーから `wandb off` を実行するか、`WANDB_MODE` 環境変数を `offline` に設定します。