---
title: SendNotification
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/actions.py#L125-L162 >}}

Defines an automation action that sends a (Slack) notification.

| Attributes |  |
| :--- | :--- |
|  `title` |  The title of the sent notification. |
|  `message` |  The message body of the sent notification. |
|  `severity` |  The severity (`INFO`, `WARN`, `ERROR`) of the sent notification. |
|  `action_type` |  The kind of action to be triggered. |

## Methods

### `from_integration`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/actions.py#L147-L162)

```python
@classmethod
from_integration(
    integration: SlackIntegration,
    *,
    title: str = "",
    text: str = "",
    level: AlertSeverity = AlertSeverity.INFO
) -> Self
```

Define a notification action that sends to the given (Slack) integration.
