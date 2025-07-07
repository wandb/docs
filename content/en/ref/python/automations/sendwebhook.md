---
title: SendWebhook
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/actions.py#L165-L187 >}}

Defines an automation action that sends a webhook request.

| Attributes |  |
| :--- | :--- |
|  `request_payload` |  The payload, possibly with template variables, to send in the webhook request. |
|  `action_type` |  The kind of action to be triggered. |

## Methods

### `from_integration`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/actions.py#L179-L187)

```python
@classmethod
from_integration(
    integration: WebhookIntegration,
    *,
    payload: Optional[SerializedToJson[dict[str, Any]]] = None
) -> Self
```

Define a webhook action that sends to the given (webhook) integration.
