---
title: グラフに何も表示されないのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-graphs_nothing_showing
support:
  - experiments
toc_hide: true
type: docs
url: /ja/support/:filename
---
"まだ可視化データがログされていません" というメッセージが表示される場合、スクリプトが最初の `wandb.log` 呼び出しを実行していないことを意味します。この状況は、run がステップを完了するのに長い時間がかかる場合に発生することがあります。データのログを迅速化するために、エポックの終わりだけでなく エポックごとに複数回ログを行ってください。