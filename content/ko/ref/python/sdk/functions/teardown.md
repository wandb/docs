---
data_type_classification: function
menu:
  reference:
    identifier: ko-ref-python-sdk-functions-teardown
object_type: python_sdk_actions
title: teardown()
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




### <kbd>function</kbd> `teardown`

```python
teardown(exit_code: 'int | None' = None) → None
```

Waits for W&B to finish and frees resources. 

Completes any runs that were not explicitly finished using `run.finish()` and waits for all data to be uploaded. 

It is recommended to call this at the end of a session that used `wandb.setup()`. It is invoked automatically in an `atexit` hook, but this is not reliable in certain setups such as when using Python's `multiprocessing` module.