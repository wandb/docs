---
title: OnRunMetric
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/events.py >}}



A run metric satisfies a user-defined condition.

Attributes:
- event_type (Literal): No description provided.
- filter (RunMetricFilter): Run and/or metric condition(s) that must be satisfied for this event to trigger an automation.
- scope (ProjectScope): The scope of the event: only projects are valid scopes for this event.

### <kbd>method</kbd> `then`
```python
then(self, action: 'InputAction') -> 'NewAutomation'
```
Define a new Automation in which this event triggers the given action.