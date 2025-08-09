---
title: OnAddArtifactAlias
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/automations/events.py#L189-L192 >}}

A new alias is assigned to an artifact.

| Attributes |  |
| :--- | :--- |
|  `scope` |  The scope of the event. |
|  `filter` |  Additional condition(s), if any, that must be met for this event to trigger an automation. |

## Methods

### `then`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.1/wandb/automations/events.py#L152-L159)

```python
then(
    action: InputAction
) -> NewAutomation
```

Define a new Automation in which this event triggers the given action.
