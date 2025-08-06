---
title: wandb.init は私のトレーニングプロセスに何をしますか？
menu:
  support:
    identifier: ja-support-kb-articles-wandbinit_training_process
support:
- 環境変数
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.init()` がトレーニングスクリプト内で実行されると、API コールによってサーバー上に run オブジェクトが作成されます。新しいプロセスが起動し、メトリクスのストリーミングと収集を行うため、メインプロセスは通常通り動作できます。スクリプトはローカルファイルに書き込みを行い、別プロセスがサーバーへデータやシステムメトリクスをストリーミングします。ストリーミングを無効にしたい場合は、トレーニングディレクトリーで `wandb off` を実行するか、`WANDB_MODE` 環境変数を `offline` に設定してください。