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

W&B はイベントをメモリーにキューイングし、非同期でディスクに書き込むことで、障害を管理し、`WANDB_MODE=offline` の設定をサポートします。これにより、ログ記録後に同期が可能になります。

ターミナル で、ローカル run ディレクトリー へのパスを確認してください。このディレクトリー には、データストアとして機能する `.wandb` ファイルが含まれています。画像 ログ記録の場合、W&B は画像をクラウドストレージにアップロードする前に `media/images` サブディレクトリー に保存します。
