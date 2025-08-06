---
title: トレーニング中にインターネット接続が切れた場合、どうなりますか？
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

ライブラリがインターネットに接続できない場合、リトライループに入り、ネットワークが復旧するまでメトリクスのストリーミングを試み続けます。この間もプログラムは実行され続けます。

インターネットに接続できないマシンで実行するには、`WANDB_MODE=offline` を設定してください。この設定により、メトリクスはローカルのハードドライブに保存されます。後で、`wandb sync DIRECTORY` を実行して、データをサーバーにストリーミングできます。