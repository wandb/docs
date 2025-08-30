---
title: OnCreateArtifact
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/automations/events.py#L195-L201 >}}

A new artifact is created.

| Attributes |  |
| :--- | :--- |
|  `scope` |  The scope of the event: only artifact collections are valid scopes for this event. |
|  `filter` |  Additional condition(s), if any, that must be met for this event to trigger an automation. |

## Methods

### `then`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/automations/events.py#L152-L159)

```python
then(
    action: InputAction
) -> NewAutomation
```

Define a new Automation in which this event triggers the given action.
