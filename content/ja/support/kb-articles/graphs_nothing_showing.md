---
title: なぜグラフに何も表示されないのですか？
menu:
  support:
    identifier: ja-support-kb-articles-graphs_nothing_showing
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

「No visualization data logged yet」というメッセージが表示される場合、スクリプトが最初の `wandb.log` 呼び出しを実行していません。この状況は、run が 1 ステップを完了するのに長い時間がかかる場合に発生することがあります。データのログを早く取得したい場合は、エポックの最後だけでなく、エポックごとに複数回ログを記録してください。