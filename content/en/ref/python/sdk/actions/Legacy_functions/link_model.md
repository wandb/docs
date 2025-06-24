---
title: link_model()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py >}}




### <kbd>function</kbd> `wandb.link_model`

```python
wandb.link_model(
    path: 'StrPath',
    registered_model_name: 'str',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → Artifact | None
```

Log a model artifact version and link it to a registered model in the model registry. 

Linked model versions are visible in the UI for the specified registered model. 

This method will: 
- Check if 'name' model artifact has been logged. If so, use the artifact version that matches the files located at 'path' or log a new version. Otherwise log files under 'path' as a new model artifact, 'name' of type 'model'. 
- Check if registered model with name 'registered_model_name' exists in the 'model-registry' project. If not, create a new registered model with name 'registered_model_name'. 
- Link version of model artifact 'name' to registered model, 'registered_model_name'. 
- Attach aliases from 'aliases' list to the newly linked model artifact version. 



**Args:**
 
 - `path`:  (str) A path to the contents of this model, can be in the  following forms: 
    - `/local/directory` 
    - `/local/directory/file.txt` 
    - `s3://bucket/path` 
 - `registered_model_name`:  The name of the registered model that the  model is to be linked to. A registered model is a collection of  model versions linked to the model registry, typically  representing a team's specific ML Task. The entity that this  registered model belongs to will be derived from the run. 
 - `name`:  The name of the model artifact that files in 'path' will be  logged to. This will default to the basename of the path  prepended with the current run id  if not specified. 
 - `aliases`:  Aliases that will only be applied on this linked artifact  inside the registered model. The alias "latest" will always be  applied to the latest version of an artifact that is linked. 



**Raises:**
 
 - `AssertionError`:  If registered_model_name is a path or  if model artifact 'name' is of a type that does not contain  the substring 'model'. 
 - `ValueError`:  If name has invalid special characters. 



**Returns:**
 The linked artifact if linking was successful, otherwise None. 
