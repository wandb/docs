---
title: 時間とともに変化しない、最終評価の精度などの指標をどのように記録できますか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- runs
toc_hide: true
type: docs
url: /ja/support/:filename
---

`wandb.log({'final_accuracy': 0.9})` を使用すると、最終精度が正しく更新されます。デフォルトでは、`wandb.log({'final_accuracy': <値>})` は `wandb.settings['final_accuracy']` を更新し、これは実行テーブルの値を反映します。