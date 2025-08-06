---
title: OnRunMetric
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-onrunmetric
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



run 메트릭이 사용자 정의 조건을 만족할 때 발생합니다.

속성:
- event_type (Literal): 설명 없음.
- filter (RunMetricFilter): 이 이벤트가 자동화(automation)를 트리거하기 위해 만족해야 하는 run 및/또는 메트릭 조건.
- scope (ProjectScope): 이벤트의 범위로, 이 이벤트에서는 Projects 만 유효한 범위입니다.

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
이 이벤트가 지정한 액션을 트리거하는 새로운 Automation을 정의합니다.
