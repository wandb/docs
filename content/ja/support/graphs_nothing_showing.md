---
title: Why is nothing showing up in my graphs?
menu:
  support:
    identifier: ja-support-graphs_nothing_showing
tags:
- experiments
toc_hide: true
type: docs
---

メッセージ「No visualization data logged yet」が表示される場合、スクリプトが最初の `wandb.log` 呼び出しを実行していない可能性があります。この状況は、run がステップを完了するのに時間がかかる場合に発生することがあります。データ ログを迅速に行うために、エポックの最後だけでなく、エポックごとに複数回ログしてください。