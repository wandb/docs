---
title: ログはトレーニングをブロックしますか？
menu:
  support:
    identifier: ja-support-kb-articles-logging_block_training
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

「ログ機能は遅延評価されていますか？ローカルでの処理中に結果をサーバーへ送信するためにネットワークに依存したくありません。」

`wandb.log` 関数は、ローカルファイルに 1 行を書き込み、ネットワーク通信で実行をブロックしません。`wandb.init` を呼び出すと、同じマシン上で新しいプロセスが起動します。このプロセスはファイルシステムの変更を監視し、Web サービスとは非同期で通信するため、ローカルの処理は中断されずに継続されます。