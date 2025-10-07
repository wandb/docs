---
title: OnLinkArtifact
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/automations/events.py#L182-L185 >}}

A new artifact is linked to a collection.

| Attributes |  |
| :--- | :--- |
|  `scope` |  The scope of the event. |
|  `filter` |  Additional condition(s), if any, that must be met for this event to trigger an automation. |

## Methods

### `then`

[View source](https://www.github.com/wandb/wandb/tree/v0.22.2/wandb/automations/events.py#L151-L158)

```python
then(
    action: InputAction
) -> NewAutomation
```

Define a new Automation in which this event triggers the given action.
