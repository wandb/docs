---
title: 웹훅 보내기
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-sendwebhook
object_type: automations_namespace
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



웹훅 요청을 보내는 자동화 액션을 정의합니다.

속성:
- action_type (Literal): 트리거될 액션의 종류입니다.
- request_payload (Optional): 웹훅 요청에 보낼 페이로드로, 템플릿 변수를 포함할 수 있습니다.

### <kbd>method</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
주어진 (웹훅) 인테그레이션으로 보낼 웹훅 액션을 정의합니다.