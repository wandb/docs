---
title: How can I log a metric that doesn't change over time such as a final evaluation
  accuracy?
menu:
  support:
    identifier: ja-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

`wandb.log({'final_accuracy': 0.9})` を使用すると、最終的な精度が正しく更新されます。デフォルトでは、`wandb.log({'final_accuracy': <value>})` は `wandb.settings['final_accuracy']` を更新し、これが runs テーブルの値を反映します。
