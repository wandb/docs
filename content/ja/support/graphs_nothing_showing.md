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

"No visualization data logged yet" （まだ可視化データがログに記録されていません）というメッセージが表示される場合、スクリプトが最初の `wandb.log` の呼び出しを実行していません。この状況は、run がステップを完了するのに長い時間がかかる場合に発生する可能性があります。データ の ログ記録を迅速化するには、エポックの最後に 1 回だけログ記録するのではなく、エポックごとに複数回ログ記録してください。
