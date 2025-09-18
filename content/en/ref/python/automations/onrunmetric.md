---
title: OnRunMetric
namespace: automations_namespace
python_object_type: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/events.py >}}



## <kbd>class</kbd> `OnRunMetric`
A run metric satisfies a user-defined condition.


### <kbd>method</kbd> `OnRunMetric.__init__`

```python
__init__(
    event_type: 'Literal[RUN_METRIC_THRESHOLD, RUN_METRIC_CHANGE]',
    scope: 'ProjectScope',
    filter: 'RunMetricFilter'
) → None
```

**Args:**
 
 - `event_type` (Literal[RUN_METRIC_THRESHOLD, RUN_METRIC_CHANGE]): 
 - `scope` (ProjectScope): The scope of the event: only projects are valid scopes for this event.
 - `filter` (RunMetricFilter): Run and/or metric condition(s) that must be satisfied for this event to trigger an automation.

**Returns:**
 An `OnRunMetric` object.
