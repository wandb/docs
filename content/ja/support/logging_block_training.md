---
title: Does logging block my training?
menu:
  support:
    identifier: ja-support-logging_block_training
tags:
- experiments
toc_hide: true
type: docs
---

"ログ関数は遅延評価されますか？ ローカルでの処理実行中に、結果 を サーバー に送信するためにネットワークに依存したくありません。"

`wandb.log` 関数は、ローカルファイルに1行書き込みますが、ネットワーク呼び出しはブロックしません。 `wandb.init` を呼び出すと、新しい プロセス が同じマシン上で開始されます。この プロセス は、ファイルシステムの変更をリッスンし、非同期的にウェブ サービス と通信するため、ローカルでの処理は中断されずに継続できます。
