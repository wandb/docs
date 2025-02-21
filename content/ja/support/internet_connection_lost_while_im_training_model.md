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

ライブラリがインターネットに接続できない場合、リトライループに入り、ネットワークが復元されるまでメトリクスのストリーミングを試み続けます。この間、プログラムは実行を続けます。

インターネットに接続されていないマシンで実行するには、`WANDB_MODE=offline` を設定してください。この設定により、メトリクスはローカルのハードドライブに保存されます。その後、`wandb sync DIRECTORY` を呼び出してデータをサーバーにストリーミングします。