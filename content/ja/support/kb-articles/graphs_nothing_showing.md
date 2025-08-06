---
title: なぜグラフに何も表示されないのですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

「No visualization data logged yet」というメッセージが表示された場合、スクリプトが最初の `wandb.log` 呼び出しを実行していません。run が各ステップの処理に時間がかかる場合、この状況が発生することがあります。データのログを早めるには、エポックの最後だけでなく、エポックごとに複数回ログを記録しましょう。