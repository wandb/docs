---
title: MetricChangeFilter
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-metricchangefilter
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}

run 메트릭의 변화가 사용자가 정의한 임계값과 비교되는 필터를 정의합니다.

변화는 "텀블링" 윈도우를 기준으로 계산되며, 즉 현재 윈도우와 이전에 겹치지 않는 윈도우 간의 차이를 의미합니다.

속성:
- agg (Optional): 윈도우 크기 내에서 적용할 수 있는 집계 연산입니다.
- change_dir (ChangeDir): 설명이 제공되지 않았습니다.
- change_type (ChangeType): 설명이 제공되지 않았습니다.
- name (str): 관찰할 메트릭의 이름입니다.
- prior_window (int): 메트릭을 집계할 이전 윈도우의 크기입니다 (`agg`가 None인 경우 무시됨).
    생략하면, 기본적으로 현재 윈도우의 크기가 사용됩니다.
- threshold (Union): 비교할 임계값입니다.
- window (int): 메트릭을 집계할 윈도우의 크기입니다 (`agg`가 None인 경우 무시됨).