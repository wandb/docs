---
title: How can I log a metric that doesn't change over time such as a final evaluation
  accuracy?
menu:
  support:
    identifier: ja-support-log_metric_doesnt_change_time_such_final
tags:
- runs
toc_hide: true
type: docs
---

`wandb.log({'final_accuracy': 0.9})` を使用すると、最終精度が正しく更新されます。デフォルトでは、`wandb.log({'final_accuracy': <値>})` は `wandb.settings['final_accuracy']` を更新し、これは runs テーブルの値を反映しています。