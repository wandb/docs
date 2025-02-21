---
title: How does wandb stream logs and writes to disk?
menu:
  support:
    identifier: ja-support-stream_logs_writes_disk
tags:
- environment variables
toc_hide: true
type: docs
---

W&B はメモリー内でイベントをキューイングし、非同期でディスクに書き込むことで障害を管理し、`WANDB_MODE=offline` 設定をサポートし、ログ記録後の同期を可能にします。

ターミナルで、ローカルな run ディレクトリーへのパスを確認してください。このディレクトリーには `.wandb` ファイルが含まれており、これはデータストアとして機能します。画像のログでは、W&B はクラウドストレージにアップロードする前に、`media/images` サブディレクトリーに画像を保存します。