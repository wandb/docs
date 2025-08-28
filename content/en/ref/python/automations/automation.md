---
title: Automation
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/v0.21.2/wandb/automations/automations.py#L19-L48 >}}

A local instance of a saved W&B automation.

| Attributes |  |
| :--- | :--- |
|  `name` |  The name of this automation. |
|  `description` |  An optional description of this automation. |
|  `enabled` |  Whether this automation is enabled. Only enabled automations will trigger. |
|  `scope` |  The scope in which the triggering event must occur. |
|  `event` |  The event that will trigger this automation. |
|  `action` |  The action that will execute when this automation is triggered. |
