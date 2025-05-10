---
title: wandb はどのようにしてログをストリームし、ディスクに書き込むのですか?
menu:
  support:
    identifier: ja-support-kb-articles-stream_logs_writes_disk
support:
  - environment variables
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: stream_logs_writes_disk
---
W&B はメモリでイベントをキューに入れ、非同期にディスクに書き込みを行って、失敗を管理し、`WANDB_MODE=offline` の設定をサポートし、ログ後の同期を可能にします。

ターミナルで、ローカルな run ディレクトリーへのパスを確認します。このディレクトリーには、データストアとして機能する `.wandb` ファイルが含まれています。画像のログでは、W&B はクラウドストレージにアップロードする前に、`media/images` サブディレクトリーに画像を保存します。