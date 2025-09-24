---
title: MetricThresholdFilter
namespace: automations_namespace
python_object_type: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



## <kbd>class</kbd> `MetricThresholdFilter`
Defines a filter that compares a run metric against a user-defined threshold value.


### <kbd>method</kbd> `MetricThresholdFilter.__init__`

```python
__init__(
    name: 'str',
    agg: 'Agg | None' = None,
    window: 'int' = 1,
    cmp: 'Literal['$gte', '$gt', '$lt', '$lte']',
    threshold: 'Annotated | Annotated'
) → None
```

**Args:**
 
 - `name` (str): 
 - `agg` (Optional[Agg]): 
 - `window` (int): 
 - `cmp` (Literal['$gte', '$gt', '$lt', '$lte']): Comparison operator used to compare the metric value (left) vs. the threshold value (right).
 - `threshold` (Union[Annotated, Annotated]): 

**Returns:**
 An `MetricThresholdFilter` object.
