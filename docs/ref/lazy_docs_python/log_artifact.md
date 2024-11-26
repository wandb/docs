import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# log_artifact

<CTAButtons githubLink='https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py'/>




### <kbd>function</kbd> `wandb.log_artifact`

```python
wandb.log_artifact(
    artifact_or_path: 'Artifact | StrPath',
    name: 'str | None' = None,
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    tags: 'list[str] | None' = None
) → Artifact
```

Declare an artifact as an output of a run. 



**Args:**
 
 - `artifact_or_path`:  (str or Artifact) A path to the contents of this artifact,  can be in the following forms: 
            - `/local/directory` 
            - `/local/directory/file.txt` 
            - `s3://bucket/path`  You can also pass an Artifact object created by calling  `wandb.Artifact`. 
 - `name`:  (str, optional) An artifact name. Valid names can be in the following forms: 
            - name:version 
            - name:alias 
            - digest  This will default to the basename of the path prepended with the current  run id  if not specified. 
 - `type`:  (str) The type of artifact to log, examples include `dataset`, `model` 
 - `aliases`:  (list, optional) Aliases to apply to this artifact,  defaults to `["latest"]` 
 - `tags`:  (list, optional) Tags to apply to this artifact, if any. 



**Returns:**
 An `Artifact` object.