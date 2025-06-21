---
title: SendWebhook
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/actions.py >}}



Defines an automation action that sends a webhook request.

Attributes:
- action_type (Literal): The kind of action to be triggered.
- request_payload (Optional): The payload, possibly with template variables, to send in the webhook request.

### <kbd>method</kbd> `from_integration`
```python
from_integration(cls, integration: 'WebhookIntegration', *, payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None) -> 'Self'
```
Define a webhook action that sends to the given (webhook) integration.