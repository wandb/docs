---
title: What happens if internet connection is lost while I'm training a model?
menu:
  support:
    identifier: ja-support-internet_connection_lost_while_im_training_model
tags:
- environment variables
toc_hide: true
type: docs
---

ライブラリ がインターネットに接続できない場合、リトライループに入り、ネットワークが復旧するまで メトリクス のストリーミングを試み続けます。この間、プログラムは実行し続けます。

インターネットに接続されていないマシンで実行するには、`WANDB_MODE=offline` を設定します。この 設定 では、 メトリクス はハードドライブにローカルに保存されます。後で、`wandb sync DIRECTORY` を呼び出して、 サーバー に データ をストリーミングします。
