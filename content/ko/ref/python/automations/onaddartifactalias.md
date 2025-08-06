---
title: OnAddArtifactAlias
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-onaddartifactalias
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



새로운 에일리어스가 아티팩트에 할당됩니다.

속성:
- event_type (Literal): 설명이 제공되지 않았습니다.
- filter (Union): 이 이벤트가 자동화를 트리거하기 위해 충족해야 하는 추가 조건(있는 경우).
- scope (Union): 이벤트의 범위.

### <kbd>메소드</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
이 이벤트가 지정된 동작을 트리거하는 새로운 Automation을 정의합니다.
