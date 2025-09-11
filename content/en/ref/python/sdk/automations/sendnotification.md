---
title: SendNotification
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



## <kbd>class</kbd> `SendNotification`
Defines an automation action that sends a (Slack) notification.


### <kbd>method</kbd> `SendNotification.__init__`

```python
__init__(
    integration_id: 'str',
    title: 'str' = '',
    message: 'str' = '',
    severity: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>,
    action_type: 'Literal[NOTIFICATION]' = NOTIFICATION
) → None
```

**Args:**
 
 - `integration_id` (str): The ID of the Slack integration that will be used to send the notification.
 - `title` (str): The title of the sent notification.
 - `message` (str): The message body of the sent notification.
 - `severity` (AlertSeverity): The severity (`INFO`, `WARN`, `ERROR`) of the sent notification.
 - `action_type` (Literal[NOTIFICATION]): 

**Returns:**
 An `SendNotification` object.

### <kbd>classmethod</kbd> `SendNotification.from_integration`

```python
from_integration(
    integration: 'SlackIntegration',
    title: 'str' = '',
    text: 'str' = '',
    level: 'AlertSeverity' = <AlertSeverity.INFO: 'INFO'>
) → Self
```

Define a notification action that sends to the given (Slack) integration.
