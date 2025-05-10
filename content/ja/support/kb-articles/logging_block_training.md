---
title: トレーニングを妨げるログはありますか？
menu:
  support:
    identifier: ja-support-kb-articles-logging_block_training
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: logging_block_training
---
" ロギング機能は遅延するのでしょうか？ローカルの操作を実行しながら、結果を サーバー に送信する際にネットワークに依存したくありません。"

`wandb.log` 関数はローカルファイルに1行を書き込み、ネットワークコールをブロックしません。`wandb.init` を呼び出すと、同じマシン上で新しいプロセスが開始されます。このプロセスはファイルシステムの変更を監視し、Web サービスと非同期で通信することで、ローカル操作が中断されることなく続行できるようにします。