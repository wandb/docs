---
title: log_model()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/lib/preinit.py >}}




### <kbd>function</kbd> `wandb.log_model`

```python
wandb.log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

Logs a model artifact as an output of this run. 

The name of model artifact can only contain alphanumeric characters, underscores, and hyphens. 



**Args:**
 
 - `path`:  (str) A path to the contents of this model,  can be in the following forms: 
    - `/local/directory` 
    - `/local/directory/file.txt` 
    - `s3://bucket/path` 
 - `name`:  A name to assign to the model artifact that  the file contents will be added to. This will default to the  basename of the path prepended with the current run id if  not specified. 
 - `aliases`:  Aliases to apply to the created model artifact,  defaults to `["latest"]` 



**Raises:**
 
 - `ValueError`:  If name has invalid special characters. 



**Returns:**
 None 



**Raises:**
 
 - `ValueError`:  if name has invalid special characters. 



**Examples:**
 ```python
run.log_model(
    path="/local/directory",
    name="my_model_artifact",
    aliases=["production"],
)
``` 

Invalid usage 

```python
run.log_model(
    path="/local/directory",
    name="my_entity/my_project/my_model_artifact",
    aliases=["production"],
)
``` 
