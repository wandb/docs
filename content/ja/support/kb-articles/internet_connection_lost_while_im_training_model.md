---
title: モデルをトレーニング中にインターネット接続が切れた場合はどうなりますか？
menu:
  support:
    identifier: ja-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- 環境変数
- 障害
toc_hide: true
type: docs
url: /support/:filename
---

ライブラリ がインターネットに接続できない場合、再試行ループに入り、ネットワークが復旧するまでメトリクスをストリーミングし続けます。この間もプログラムは実行し続けます。

インターネットに接続できないマシンで実行するには、`WANDB_MODE=offline` を設定します。この 設定 はメトリクスをハードドライブ上にローカル保存します。後で、`wandb sync DIRECTORY` を実行して、データを サーバー にストリーミングします。