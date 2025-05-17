---
title: Class SendWebhook
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/actions.py >}}




## <kbd>class</kbd> `SendWebhook`
Defines an automation action that sends a webhook request. 


---

### <kbd>property</kbd> SendWebhook.model_extra

Get extra fields set during validation. 



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`. 

---

### <kbd>property</kbd> SendWebhook.model_fields_set

Returns the set of fields that have been explicitly set on this model instance. 



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults. 



---

### <kbd>classmethod</kbd> `SendWebhook.from_integration`

```python
from_integration(
    integration: 'WebhookIntegration',
    payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None
) → Self
```

Define a webhook action that sends to the given (webhook) integration. 

