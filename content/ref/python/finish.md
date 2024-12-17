---
title: finish
---

{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/sdk/wandb_run.py#L4140-L4151">}}

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
