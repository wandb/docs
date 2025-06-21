---
title: OnCreateArtifact
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/events.py >}}



A new artifact is created.

Attributes:
- event_type (Literal): No description provided.
- filter (Union): Additional condition(s), if any, that must be met for this event to trigger an automation.
- scope (Union): The scope of the event: only artifact collections are valid scopes for this event.

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
Define a new Automation in which this event triggers the given action.