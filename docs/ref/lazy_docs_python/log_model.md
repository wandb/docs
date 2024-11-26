import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# log_model

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.log_model`

```python
wandb.log_model(
    path: 'StrPath',
    name: 'str | None' = None,
    aliases: 'list[str] | None' = None
) → None
```

Logs a model artifact containing the contents inside the 'path' to a run and marks it as an output to this run. 



**Args:**
 
 - `path`:  (str) A path to the contents of this model,  can be in the following forms: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path` 
 - `name`:  (str, optional) A name to assign to the model artifact that the file contents will be added to. 
 - `The string must contain only the following alphanumeric characters`:  dashes, underscores, and dots. This will default to the basename of the path prepended with the current run id  if not specified. 
 - `aliases`:  (list, optional) Aliases to apply to the created model artifact,  defaults to `["latest"]` 



**Examples:**
 ```python
    run.log_model(
         path="/local/directory",
         name="my_model_artifact",
         aliases=["production"],
    )
    ``` 

Invalid usage ```python
    run.log_model(
         path="/local/directory",
         name="my_entity/my_project/my_model_artifact",
         aliases=["production"],
    )
    ``` 



**Raises:**
 
 - `ValueError`:  if name has invalid special characters 



**Returns:**
 None