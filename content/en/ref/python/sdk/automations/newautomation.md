---
title: NewAutomation
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/automations.py >}}



## <kbd>class</kbd> `NewAutomation`
A new automation to be created.


### <kbd>method</kbd> `NewAutomation.__init__`

```python
__init__(
    name: 'str | None' = None,
    description: 'str | None' = None,
    enabled: 'bool | None' = None,
    event: 'Annotated | None' = None,
    action: 'Annotated | None' = None
) → None
```

**Args:**
 
 - `name` (Optional[str]): The name of this automation.
 - `description` (Optional[str]): An optional description of this automation.
 - `enabled` (Optional[bool]): Whether this automation is enabled.  Only enabled automations will trigger.
 - `event` (Optional[Annotated]): The event that will trigger this automation.
 - `action` (Optional[Annotated]): The action that will execute when this automation is triggered.

**Returns:**
 An `NewAutomation` object.

### <kbd>property</kbd> `NewAutomation.scope`

The scope in which the triggering event must occur.

**Returns:**
 - `Optional[AutomationScope]`: The scope property value.
