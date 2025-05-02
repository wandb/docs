---
title: How can I log a metric that doesn't change over time such as a final evaluation
  accuracy?
menu:
  support:
    identifier: ko-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

`wandb.log({'final_accuracy': 0.9})`를 사용하면 최종 정확도가 올바르게 업데이트됩니다. 기본적으로 `wandb.log({'final_accuracy': <값>})`는 `wandb.settings['final_accuracy']`를 업데이트하며, 이는 runs table의 값을 반영합니다.
