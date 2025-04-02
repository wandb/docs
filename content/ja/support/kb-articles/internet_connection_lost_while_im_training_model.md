---
title: What happens if internet connection is lost while I'm training a model?
menu:
  support:
    identifier: ja-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

ライブラリがインターネットに接続できない場合、再試行ループに入り、ネットワークが復旧するまでメトリクスのストリーミングを試行し続けます。この間、プログラムは実行し続けます。

インターネットに接続されていないマシンで実行するには、`WANDB_MODE=offline` を設定します。この設定では、メトリクスはハードドライブにローカルに保存されます。後で、`wandb sync DIRECTORY` を呼び出して、データをサーバーにストリーミングします。
