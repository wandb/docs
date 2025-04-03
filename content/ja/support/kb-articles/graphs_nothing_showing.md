---
title: Why is nothing showing up in my graphs?
menu:
  support:
    identifier: ja-support-kb-articles-graphs_nothing_showing
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

「No visualization data logged yet」というメッセージが表示される場合、スクリプトが最初の `wandb.log` の呼び出しを実行していません。この状況は、run がステップを完了するのに長い時間がかかる場合に発生することがあります。データ の ログ を迅速化するには、エポック の最後にのみ ログ を記録するのではなく、エポック ごとに複数回 ログ を記録します。
