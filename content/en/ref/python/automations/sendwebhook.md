---
title: SendWebhook
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/automations/actions.py#L167-L189 >}}

Defines an automation action that sends a webhook request.

| Attributes |  |
| :--- | :--- |
|  `request_payload` |  The payload, possibly with template variables, to send in the webhook request. |
|  `action_type` |  The kind of action to be triggered. |

## Methods

### `from_integration`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/automations/actions.py#L181-L189)

```python
@classmethod
from_integration(
    integration: WebhookIntegration,
    *,
    payload: Optional[SerializedToJson[dict[str, Any]]] = None
) -> Self
```

Define a webhook action that sends to the given (webhook) integration.
