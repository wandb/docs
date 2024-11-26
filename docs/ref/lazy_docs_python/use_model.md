import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# use_model

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.use_model`

```python
wandb.use_model(name: 'str') → FilePathStr
```

Download the files logged in a model artifact 'name'. 



**Args:**
 
 - `name`:  (str) A model artifact name. 'name' must match the name of an existing logged  model artifact.  May be prefixed with entity/project/. Valid names  can be in the following forms: 
            - model_artifact_name:version 
            - model_artifact_name:alias 



**Examples:**
 ```python
    run.use_model(
         name="my_model_artifact:latest",
    )

    run.use_model(
         name="my_project/my_model_artifact:v0",
    )

    run.use_model(
         name="my_entity/my_project/my_model_artifact:<digest>",
    )
    ``` 

Invalid usage ```python
    run.use_model(
         name="my_entity/my_project/my_model_artifact",
    )
    ``` 



**Raises:**
 
 - `AssertionError`:  if model artifact 'name' is of a type that does not contain the substring 'model'. 



**Returns:**
 
 - `path`:  (str) path to downloaded model artifact file(s).