---
title: MetricChangeFilter
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/_filters/run_metrics.py >}}




## <kbd>class</kbd> `MetricChangeFilter`
Defines a filter that compares a change in a run metric against a user-defined threshold. 

The change is calculated over "tumbling" windows, i.e. the difference between the current window and the non-overlapping prior window. 


---

### <kbd>property</kbd> MetricChangeFilter.model_extra

Get extra fields set during validation. 



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`. 

---

### <kbd>property</kbd> MetricChangeFilter.model_fields_set

Returns the set of fields that have been explicitly set on this model instance. 



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults. 



