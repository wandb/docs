---
data_type_classification: class
menu:
  reference:
    identifier: ja-ref-python-automations-onaddartifactalias
object_type: automations_namespace
title: OnAddArtifactAlias
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



A new alias is assigned to an artifact.

Attributes:
- event_type (Literal): No description provided.
- filter (Union): Additional condition(s), if any, that must be met for this event to trigger an automation.
- scope (Union): The scope of the event.

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
Define a new Automation in which this event triggers the given action.