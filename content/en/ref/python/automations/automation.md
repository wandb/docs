---
title: Automation
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/automations.py >}}



A local instance of a saved W&B automation.

Attributes:
- action (Union): The action that will execute when this automation is triggered.
- description (Optional): An optional description of this automation.
- enabled (bool): Whether this automation is enabled.  Only enabled automations will trigger.
- event (SavedEvent): The event that will trigger this automation.
- name (str): The name of this automation.
- scope (Union): The scope in which the triggering event must occur.