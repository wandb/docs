---
title: OnCreateArtifact
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/automations/events.py >}}




## <kbd>class</kbd> `OnCreateArtifact`
A new artifact is created. 


---

### <kbd>property</kbd> OnCreateArtifact.model_extra

Get extra fields set during validation. 



**Returns:**
  A dictionary of extra fields, or `None` if `config.extra` is not set to `"allow"`. 

---

### <kbd>property</kbd> OnCreateArtifact.model_fields_set

Returns the set of fields that have been explicitly set on this model instance. 



**Returns:**
  A set of strings representing the fields that have been set,  i.e. that were not filled from defaults. 



---

### <kbd>method</kbd> `OnCreateArtifact.then`

```python
then(action: 'InputAction') → NewAutomation
```

Define a new Automation in which this event triggers the given action. 

