---
title: OnRunMetric
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/events.py#L211-L233 >}}

A run metric satisfies a user-defined condition.

| Attributes |  |
| :--- | :--- |
|  `scope` |  The scope of the event: only projects are valid scopes for this event. |
|  `filter` |  Run and/or metric condition(s) that must be satisfied for this event to trigger an automation. |

## Methods

### `then`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/events.py#L152-L159)

```python
then(
    action: InputAction
) -> NewAutomation
```

Define a new Automation in which this event triggers the given action.
