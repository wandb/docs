---
title: SendWebhook
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/actions.py >}}



## <kbd>class</kbd> `SendWebhook`
Defines an automation action that sends a webhook request.


**Args:**
 
 - `integration_id` (str): The ID of the webhook integration that will be used to send the request.
 - `request_payload` (Optional[Annotated]): The payload, possibly with template variables, to send in the webhook request.
 - `action_type` (Literal[GENERIC_WEBHOOK]): 

**Returns:**
 An `SendWebhook` object.

### <kbd>method</kbd> `SendWebhook.__init__`

```python
__init__(
    integration_id: 'str',
    request_payload: 'Annotated | None' = None,
    action_type: 'Literal[GENERIC_WEBHOOK]' = GENERIC_WEBHOOK
) → None
```

### <kbd>classmethod</kbd> `SendWebhook.from_integration`

```python
from_integration(
    integration: 'WebhookIntegration',
    payload: 'Optional[SerializedToJson[dict[str, Any]]]' = None
) → Self
```

Define a webhook action that sends to the given (webhook) integration.
