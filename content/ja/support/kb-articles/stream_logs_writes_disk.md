---
title: wandb はどのようにログをストリーミングし、ディスクに書き込むのですか？
menu:
  support:
    identifier: ja-support-kb-articles-stream_logs_writes_disk
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

W&B はイベントをメモリ上でキューし、ディスクへ非同期で書き込むことで障害に対応し、`WANDB_MODE=offline` の設定をサポートします。これにより、ログ取得後に同期が可能です。

ターミナルでローカル run ディレクトリーへのパスを確認できます。このディレクトリーにはデータストアとして機能する `.wandb` ファイルが含まれています。画像ログの場合、W&B は画像を `media/images` サブディレクトリーに保存し、その後クラウドストレージにアップロードします。