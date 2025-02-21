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

`wandb.init()` がトレーニングスクリプト で実行されると、API 呼び出しによってサーバー上に run オブジェクト が作成されます。新しい プロセス が開始され、 メトリクス のストリーミングと収集が行われ、メインの プロセス が正常に機能するようになります。スクリプト はローカルファイルに書き込み、別の プロセス がシステム メトリクス を含むデータを サーバー にストリーミングします。ストリーミングをオフにするには、トレーニング ディレクトリー から `wandb off` を実行するか、`WANDB_MODE` 環境 変数 を `offline` に設定します。
