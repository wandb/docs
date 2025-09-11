---
title: MetricChangeFilter
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



## <kbd>class</kbd> `MetricChangeFilter`
Defines a filter that compares a change in a run metric against a user-defined threshold.

The change is calculated over "tumbling" windows, i.e. the difference
between the current window and the non-overlapping prior window.


### <kbd>method</kbd> `MetricChangeFilter.__init__`

```python
__init__(
    name: 'str',
    agg: 'Agg | None' = None,
    window: 'int' = 1,
    cmp: 'None' = None,
    threshold: 'Annotated | Annotated',
    prior_window: 'int' = None,
    change_type: 'ChangeType',
    change_dir: 'ChangeDir'
) → None
```

**Args:**
 
 - `name` (str): 
 - `agg` (Optional[Agg]): 
 - `window` (int): 
 - `cmp` (None): Ignored.
 - `threshold` (Union[Annotated, Annotated]): 
 - `prior_window` (int): Size of the prior window over which the metric is aggregated (ignored if `agg is None`).
   If omitted, defaults to the size of the current window.
 - `change_type` (ChangeType): 
 - `change_dir` (ChangeDir): 

**Returns:**
 An `MetricChangeFilter` object.
