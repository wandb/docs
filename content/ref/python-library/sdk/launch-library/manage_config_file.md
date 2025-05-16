---
title: manage_config_file()
object_type: launch_apis_namespace
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/launch/inputs/manage.py >}}




### <kbd>function</kbd> `manage_config_file`

```python
manage_config_file(
    path: str,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    schema: Optional[Any] = None
)
```

Declare an overridable configuration file for a launch job. 

If a new job version is created from the active run, the configuration file will be added to the job's inputs. If the job is launched and overrides have been provided for the configuration file, this function will detect the overrides from the environment and update the configuration file on disk. Note that these overrides will only be applied in ephemeral containers. `include` and `exclude` are lists of dot separated paths with the config. The paths are used to filter subtrees of the configuration file out of the job's inputs. 

For example, given the following configuration file: ```yaml
     model:
         name: resnet
         layers: 18
     training:
         epochs: 10
         batch_size: 32
    ``` 

Passing `include=['model']` will only include the `model` subtree in the job's inputs. Passing `exclude=['model.layers']` will exclude the `layers` key from the `model` subtree. Note that `exclude` takes precedence over `include`. 

`.` is used as a separator for nested keys. If a key contains a `.`, it should be escaped with a backslash, e.g. `include=[r'model\.layers']`. Note the use of `r` to denote a raw string when using escape chars. 



**Args:**
 
 - `path` (str):  The path to the configuration file. This path must be  relative and must not contain backwards traversal, i.e. `..`. 
 - `include` (List[str]):  A list of keys to include in the configuration file. 
 - `exclude` (List[str]):  A list of keys to exclude from the configuration file. 
 - `schema` (dict | Pydantic model):  A JSON Schema or Pydantic model describing  describing which attributes will be editable from the Launch drawer.  Accepts both an instance of a Pydantic BaseModel class or the BaseModel  class itself. 



**Raises:**
 
 - `LaunchError`:  If the path is not valid, or if there is no active run. 
