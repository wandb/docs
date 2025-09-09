---
title: ログの記録は トレーニング を妨げますか？
menu:
  support:
    identifier: ja-support-kb-articles-logging_block_training
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

「ログ関数は遅延的ですか？ ローカルの処理を実行しているあいだに、ネットワークに依存してサーバーに結果を送信したくありません。」

`wandb.log` はローカルファイルに1行を書き込み、ネットワーク呼び出しをブロックしません。`wandb.init` を呼び出すと、同じマシン上で新しいプロセスが開始されます。このプロセスはファイルシステムの変更を監視し、Web サービスと非同期に通信するため、ローカルの処理は中断されることなく続行できます。