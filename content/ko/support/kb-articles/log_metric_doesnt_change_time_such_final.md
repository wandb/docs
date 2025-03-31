---
menu:
  support:
    identifier: ko-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- runs
title: How can I log a metric that doesn't change over time such as a final evaluation
  accuracy?
toc_hide: true
type: docs
url: /support/:filename
---

Using `wandb.log({'final_accuracy': 0.9})` updates the final accuracy correctly. By default, `wandb.log({'final_accuracy': <value>})` updates `wandb.settings['final_accuracy']`, which reflects the value in the runs table.