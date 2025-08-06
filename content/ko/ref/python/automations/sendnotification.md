---
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-sendnotification
object_type: automations_namespace
title: SendNotification
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



Defines an automation action that sends a (Slack) notification.

Attributes:
- action_type (Literal): The kind of action to be triggered.
- message (str): The message body of the sent notification.
- severity (AlertSeverity): The severity (`INFO`, `WARN`, `ERROR`) of the sent notification.
- title (str): The title of the sent notification.

### <kbd>method</kbd> `from_integration`
```python
from_integration(cls, integration: 'SlackIntegration', *, title: 'str' = '', text: 'str' = '', level: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>) -> 'Self'
```
Define a notification action that sends to the given (Slack) integration.