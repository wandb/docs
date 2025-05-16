---
title: log_artifact()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/lib/preinit.py >}}




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
 
 - `artifact_or_path`:  A path to the contents of this artifact,  can be in the following forms 
    - `/local/directory` 
    - `/local/directory/file.txt` 
    - `s3://bucket/path` 
 - `name`:  An artifact name. Defaults to the basename of the path  prepended with the current run id if not specified.  Valid names can be in the following forms: 
    - name:version 
    - name:alias 
    - digest 
 - `type`:  The type of artifact to log. Common examples include  `dataset` and `model` 
 - `aliases`:  Aliases to apply to this artifact,  defaults to `["latest"]` 
 - `tags`:  Tags to apply to this artifact, if any. 



**Returns:**
 An `Artifact` object. 
