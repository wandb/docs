---
title: SendNotification
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/actions.py >}}




## <kbd>class</kbd> `SendNotification`
Defines an automation action that sends a (Slack) notification. 


---

### <kbd>property</kbd> SendNotification.model_extra

Get extra fields set during validation. 



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`. 

---

### <kbd>property</kbd> SendNotification.model_fields_set

Returns the set of fields that have been explicitly set on this model instance. 



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults. 



---

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

