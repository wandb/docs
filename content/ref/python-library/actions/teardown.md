---
title: teardown
object_type: api
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/wandb_setup.py >}}




### <kbd>function</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

Waits for wandb to finish and frees resources. 

Completes any runs that were not explicitly finished using `run.finish()` and waits for all data to be uploaded. 

It is recommended to call this at the end of a session that used `wandb.setup()`. It is invoked automatically in an `atexit` hook, but this is not reliable in certain setups such as when using Python's `multiprocessing` module. 
