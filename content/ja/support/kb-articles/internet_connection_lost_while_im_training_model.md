---
title: モデルのトレーニング中にインターネット接続が切れた場合、何が起こりますか？
menu:
  support:
    identifier: ja-support-kb-articles-internet_connection_lost_while_im_training_model
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

ライブラリがインターネットに接続できない場合、再試行ループに入り、ネットワークが復旧するまでメトリクスのストリームを試み続けます。この間、プログラムは実行を継続します。

インターネットなしでマシン上で実行するには、`WANDB_MODE=offline` を設定します。この設定は、メトリクスをローカルのハードドライブに保存します。その後、`wandb sync ディレクトリー` を呼び出して、データをサーバーにストリームします。