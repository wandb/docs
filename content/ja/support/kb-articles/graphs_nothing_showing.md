---
title: グラフに何も表示されないのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-graphs_nothing_showing
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

「No visualization data logged yet」というメッセージが表示される場合、スクリプトが最初の `wandb.log` の呼び出しを実行していません。この状況は、 run がステップを完了するのに時間がかかっているときに発生することがあります。データのログを早めるには、最後に 1 回だけではなく、エポックごとに複数回ログしてください。