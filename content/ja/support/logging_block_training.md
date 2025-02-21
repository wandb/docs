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

"ログ機能は遅延しているのですか？ローカル操作を実行しながら、ネットワークに依存して結果をサーバーに送信することは避けたいです。"

`wandb.log` 関数はローカルファイルに行を書き込み、ネットワークコールをブロックしません。`wandb.init` を呼び出すと、同じマシンで新しいプロセスが開始されます。このプロセスはファイルシステムの変更を監視し、ウェブサービスと非同期で通信することで、ローカル操作が中断されることなく続行できるようにします。