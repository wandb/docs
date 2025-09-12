---
title: OnAddArtifactAlias
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/automations/events.py#L188-L191 >}}

A new alias is assigned to an artifact.

| Attributes |  |
| :--- | :--- |
|  `scope` |  The scope of the event. |
|  `filter` |  Additional condition(s), if any, that must be met for this event to trigger an automation. |

## Methods

### `then`

[View source](https://www.github.com/wandb/wandb/tree/v0.21.4/wandb/automations/events.py#L151-L158)

```python
then(
    action: InputAction
) -> NewAutomation
```

Define a new Automation in which this event triggers the given action.
