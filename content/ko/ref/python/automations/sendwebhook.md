---
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-sendwebhook
object_type: automations_namespace
title: SendWebhook
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



Defines an automation action that sends a webhook request.

Attributes:
- action_type (Literal): The kind of action to be triggered.
- request_payload (Optional): The payload, possibly with template variables, to send in the webhook request.

### <kbd>method</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
Define a webhook action that sends to the given (webhook) integration.