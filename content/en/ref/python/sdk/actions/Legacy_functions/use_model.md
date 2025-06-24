---
title: use_model()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py >}}




### <kbd>function</kbd> `wandb.use_model`

```python
wandb.use_model(name: 'str') → FilePathStr
```

Download the files logged in a model artifact 'name'. 



**Args:**
 
 - `name`:  A model artifact name. 'name' must match the name of an existing logged  model artifact. May be prefixed with `entity/project/`. Valid names  can be in the following forms 
    - model_artifact_name:version 
    - model_artifact_name:alias 



**Raises:**
 
 - `AssertionError`:  If model artifact 'name' is of a type that does not contain the substring 'model'. 



**Returns:**
 
 - `path` (str):  Path to downloaded model artifact file(s). 
