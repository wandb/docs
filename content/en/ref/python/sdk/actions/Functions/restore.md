---
title: restore()
object_type: python_sdk_actions
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py >}}




### <kbd>function</kbd> `restore`

```python
restore(
    name: 'str',
    run_path: 'str | None' = None,
    replace: 'bool' = False,
    root: 'str | None' = None
) → None | TextIO
```

Download the specified file from cloud storage. 

File is placed into the current directory or run directory. By default, will only download the file if it doesn't already exist. 



**Args:**
 
 - `name`:  The name of the file. 
 - `run_path`:  Optional path to a run to pull files from, i.e. `username/project_name/run_id`  if wandb.init has not been called, this is required. 
 - `replace`:  Whether to download the file even if it already exists locally 
 - `root`:  The directory to download the file to.  Defaults to the current  directory or the run directory if wandb.init was called. 



**Returns:**
 None if it can't find the file, otherwise a file object open for reading. 



**Raises:**
 
 - `wandb.CommError`:  If W&B can't connect to the W&B backend. 
 - `ValueError`:  If the file is not found or can't find run_path. 
