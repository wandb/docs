---
title: wandb.init はトレーニング プロセスに何を行いますか？
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

トレーニングスクリプトで `wandb.init()` が実行されると、API 呼び出しによってサーバー上に run オブジェクトが作成されます。新しいプロセスがメトリクスのストリーミングと収集を開始し、メインプロセスが通常どおり動作できるようにします。スクリプトはローカルファイルに書き込み、別のプロセスがシステムメトリクスを含むデータをサーバーにストリーミングします。ストリーミングをオフにするには、トレーニングディレクトリーで `wandb off` を実行するか、`WANDB_MODE` 環境変数を `offline` に設定します。