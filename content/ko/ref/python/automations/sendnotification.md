---
title: SendNotification
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-sendnotification
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



(슬랙) 알림을 보내는 오토메이션 액션을 정의합니다.

속성:
- action_type (Literal): 트리거될 액션의 종류입니다.
- message (str): 전송되는 알림의 본문 메시지입니다.
- severity (AlertSeverity): 전송되는 알림의 심각도 (`INFO`, `WARN`, `ERROR`)입니다.
- title (str): 전송되는 알림의 제목입니다.

### <kbd>메소드</kbd> `from_integration`
```python
from_integration(cls, integration: 'SlackIntegration', *, title: 'str' = '', text: 'str' = '', level: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>) -> 'Self'
```
지정한 (Slack) 인테그레이션으로 알림을 보내는 액션을 정의합니다.
