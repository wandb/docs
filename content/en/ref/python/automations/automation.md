---
title: Automation
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/automations/automations.py#L19-L48 >}}

A local instance of a saved W&B automation.

| Attributes |  |
| :--- | :--- |
|  `name` |  The name of this automation. |
|  `description` |  An optional description of this automation. |
|  `enabled` |  Whether this automation is enabled. Only enabled automations will trigger. |
|  `scope` |  The scope in which the triggering event must occur. |
|  `event` |  The event that will trigger this automation. |
|  `action` |  The action that will execute when this automation is triggered. |
