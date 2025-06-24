---
title: use_artifact()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/lib/preinit.py >}}




### <kbd>function</kbd> `wandb.use_artifact`

```python
wandb.use_artifact(
    artifact_or_name: 'str | Artifact',
    type: 'str | None' = None,
    aliases: 'list[str] | None' = None,
    use_as: 'str | None' = None
) → Artifact
```

Declare an artifact as an input to a run. 

Call `download` or `file` on the returned object to get the contents locally. 



**Args:**
 
 - `artifact_or_name`:  The name of the artifact to use. May be prefixed  with the name of the project the artifact was logged to  ("<entity>" or "<entity>/<project>"). If no  entity is specified in the name, the Run or API setting's entity is used.  Valid names can be in the following forms 
    - name:version 
    - name:alias 
 - `type`:  The type of artifact to use. 
 - `aliases`:  Aliases to apply to this artifact 
 - `use_as`:  This argument is deprecated and does nothing. 



**Returns:**
 An `Artifact` object. 



**Examples:**
 ```python
import wandb

run = wandb.init(project="<example>")

# Use an artifact by name and alias
artifact_a = run.use_artifact(artifact_or_name="<name>:<alias>")

# Use an artifact by name and version
artifact_b = run.use_artifact(artifact_or_name="<name>:v<version>")

# Use an artifact by entity/project/name:alias
artifact_c = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:<alias>"
)

# Use an artifact by entity/project/name:version
artifact_d = run.use_artifact(
    artifact_or_name="<entity>/<project>/<name>:v<version>"
)
``` 
