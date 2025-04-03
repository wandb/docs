---
title: How does wandb stream logs and writes to disk?
menu:
  support:
    identifier: ja-support-kb-articles-stream_logs_writes_disk
support:
- environment variables
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、イベントをメモリーにキューに入れ、非同期でディスクに書き込むことで、障害を管理し、`WANDB_MODE=offline` 設定をサポートします。これにより、ログ記録後に同期が可能になります。

ターミナル で、ローカル の run ディレクトリー へのパスを確認します。このディレクトリー には、データストア として機能する `.wandb` ファイルが含まれています。画像ログ の場合、W&B は画像をクラウドストレージ にアップロードする前に、`media/images` サブディレクトリー に保存します。
