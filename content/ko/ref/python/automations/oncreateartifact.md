---
title: OnCreateArtifact
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-oncreateartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



새로운 artifact 가 생성됩니다.

속성:
- event_type (Literal): 설명이 제공되지 않았습니다.
- filter (Union): 이 이벤트가 automation 을 트리거하기 위해 충족되어야 할 추가 조건(있는 경우)입니다.
- scope (Union): 이벤트의 범위입니다. 이 이벤트에 대해 유효한 범위는 오직 artifact 컬렉션입니다.

### <kbd>메소드</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
이 이벤트가 주어진 action 을 트리거하는 새로운 Automation 을 정의합니다.
