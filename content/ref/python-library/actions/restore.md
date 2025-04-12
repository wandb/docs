---
title: restore
object_type: api
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_run.py >}}




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
 
 - `name`:  the name of the file 
 - `run_path`:  Path to a run to pull files from  `username/project_name/run_id`. If `wandb.init`  has not been called, this is required. 
 - `replace`:  whether to download the file even if it already exists locally 
 - `root`:  the directory to download the file to.  Defaults to the current  directory or the run directory if `wandb.init` was called. 



**Returns:**
 None if it can't find the file, otherwise a file object open for reading 



**Raises:**
 
 - `wandb.CommError`:  if we can't connect to the wandb backend 
 - `ValueError`:  if the file is not found or can't find run_path 
