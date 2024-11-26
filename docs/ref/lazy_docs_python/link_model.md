import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# link_model

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.link_model`

```python
wandb.link_model(
    path: 'StrPath',
    registered_model_name: 'str',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

Log a model artifact version and link it to a registered model in the model registry. 

The linked model version will be visible in the UI for the specified registered model. 

Steps: 
    - Check if 'name' model artifact has been logged. If so, use the artifact version that matches the files  located at 'path' or log a new version. Otherwise log files under 'path' as a new model artifact, 'name'  of type 'model'. 
    - Check if registered model with name 'registered_model_name' exists in the 'model-registry' project.  If not, create a new registered model with name 'registered_model_name'. 
    - Link version of model artifact 'name' to registered model, 'registered_model_name'. 
    - Attach aliases from 'aliases' list to the newly linked model artifact version. 



**Args:**
 
 - `path`:  (str) A path to the contents of this model,  can be in the following forms: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path` 
 - `registered_model_name`:  (str) - the name of the registered model that the model is to be linked to.  A registered model is a collection of model versions linked to the model registry, typically representing a  team's specific ML Task. The entity that this registered model belongs to will be derived from the run 
 - `name`:  (str, optional) - the name of the model artifact that files in 'path' will be logged to. This will  default to the basename of the path prepended with the current run id  if not specified. 
 - `aliases`:  (List[str], optional) - alias(es) that will only be applied on this linked artifact  inside the registered model.  The alias "latest" will always be applied to the latest version of an artifact that is linked. 



**Examples:**
 ```python
    run.link_model(
         path="/local/directory",
         registered_model_name="my_reg_model",
         name="my_model_artifact",
         aliases=["production"],
    )
    ``` 

Invalid usage 
```python
run.link_model(
        path="/local/directory",
        registered_model_name="my_entity/my_project/my_reg_model",
        name="my_model_artifact",
        aliases=["production"],
)

run.link_model(
        path="/local/directory",
        registered_model_name="my_reg_model",
        name="my_entity/my_project/my_model_artifact",
        aliases=["production"],
)
    ``` 



**Raises:**
 
 - `AssertionError`:  if registered_model_name is a path or  if model artifact 'name' is of a type that does not contain the substring 'model' 
 - `ValueError`:  if name has invalid special characters 



**Returns:**
 None