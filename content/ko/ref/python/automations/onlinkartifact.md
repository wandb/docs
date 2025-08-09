---
title: OnLinkArtifact
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-onlinkartifact
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



새로운 artifact 가 컬렉션에 연결됩니다.

속성:
- event_type (Literal): 설명이 제공되지 않았습니다.
- filter (Union): 이 이벤트가 automation 을 트리거하려면 충족해야 하는 추가 조건(있는 경우)입니다.
- scope (Union): 이벤트의 범위입니다.

### <kbd>메소드</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
이 이벤트가 지정된 action 을 트리거하도록 하는 새로운 Automation 을 정의합니다.
