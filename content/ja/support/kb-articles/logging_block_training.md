---
title: Does logging block my training?
menu:
  support:
    identifier: ja-support-kb-articles-logging_block_training
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

「ログ関数は遅延評価されますか？ ローカル処理 の実行中に、 サーバー に 結果 を送信するためにネットワークに依存したくありません。」

`wandb.log` 関数は、ローカルファイルに1行書き込みますが、ネットワーク呼び出しはブロックしません。`wandb.init` を呼び出すと、新しい プロセス が同じマシン上で開始されます。この プロセス はファイルシステムの変更を監視し、ウェブ サービス と非同期的に通信することで、ローカル処理 が中断されることなく続行されます。
