---
title: 'MetricThresholdFilter

  '
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-metricthresholdfilter
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



run 메트릭을 사용자 정의 임계값과 비교하는 필터를 정의합니다.

속성:
- agg (Optional): 적용할 집계(aggregate) 연산. 윈도우 크기 내에서만 사용됩니다.
- cmp (Literal): 메트릭 값(왼쪽)과 임계값(오른쪽)을 비교할 때 사용하는 비교 연산자.
- name (str): 관찰할 메트릭의 이름.
- threshold (Union): 비교할 임계값.
- window (int): 메트릭을 집계할 때 사용할 윈도우 크기 (`agg`가 None이면 무시됨).