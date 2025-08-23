---
title: 시간에 따라 변하지 않는 최종 평가 정확도와 같은 metric을 어떻게 로그할 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

`run.log({'final_accuracy': 0.9})` 를 사용하면 최종 정확도가 올바르게 업데이트됩니다. 기본적으로 `run.log({'final_accuracy': <value>})` 는 `run.settings['final_accuracy']` 를 업데이트하며, 이는 Runs 테이블의 값을 반영합니다.