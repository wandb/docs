---
title: finish
---
<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/wandb_run.py#L4140-L4151' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


Mark a run as finished, and finish uploading all data.

```python
finish(
    exit_code: (int | None) = None,
    quiet: (bool | None) = None
) -> None
```

This is used when creating multiple runs in the same process.
We automatically call this method when your script exits.

| Args |  |
| :--- | :--- |
|  `exit_code` |  Set to something other than 0 to mark a run as failed |
|  `quiet` |  Deprecated, use `wandb.Settings(quiet=...)` to set this instead. |