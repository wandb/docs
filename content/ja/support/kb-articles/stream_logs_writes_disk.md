---
title: wandb はどのようにログをストリーミングし、ディスクに書き込みますか？
menu:
  support:
    identifier: ja-support-kb-articles-stream_logs_writes_disk
support:
- 環境変数
toc_hide: true
type: docs
url: /support/:filename
---

W&B は、障害の管理と `WANDB_MODE=offline` 設定のサポートのために、イベントをメモリにキューし、非同期でディスクに書き込みます。これにより、ログ後に同期できます。

ターミナルで、ローカルの run ディレクトリーへのパスを確認します。このディレクトリーにはデータストアとなる `.wandb` ファイルが含まれます。画像のログの場合、W&B はクラウド ストレージにアップロードする前に `media/images` サブディレクトリーに画像を保存します。