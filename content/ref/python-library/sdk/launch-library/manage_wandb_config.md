---
title: manage_wandb_config()
object_type: launch_apis_namespace
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/launch/inputs/manage.py >}}




### <kbd>function</kbd> `manage_wandb_config`

```python
manage_wandb_config(
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    schema: Optional[Any] = None
)
```

Declare wandb.config as an overridable configuration for a launch job. 

If a new job version is created from the active run, the run config (wandb.config) will become an overridable input of the job. If the job is launched and overrides have been provided for the run config, the overrides will be applied to the run config when `wandb.init` is called. `include` and `exclude` are lists of dot separated paths with the config. The paths are used to filter subtrees of the configuration file out of the job's inputs. 

For example, given the following run config contents: ```yaml
     model:
         name: resnet
         layers: 18
     training:
         epochs: 10
         batch_size: 32
    ``` Passing `include=['model']` will only include the `model` subtree in the job's inputs. Passing `exclude=['model.layers']` will exclude the `layers` key from the `model` subtree. Note that `exclude` takes precedence over `include`. `.` is used as a separator for nested keys. If a key contains a `.`, it should be escaped with a backslash, e.g. `include=[r'model\.layers']`. Note the use of `r` to denote a raw string when using escape chars. 



**Args:**
 
 - `include` (List[str]):  A list of subtrees to include in the configuration. 
 - `exclude` (List[str]):  A list of subtrees to exclude from the configuration. 
 - `schema` (dict | Pydantic model):  A JSON Schema or Pydantic model describing  describing which attributes will be editable from the Launch drawer.  Accepts both an instance of a Pydantic BaseModel class or the BaseModel  class itself. 



**Raises:**
 
 - `LaunchError`:  If there is no active run. 
