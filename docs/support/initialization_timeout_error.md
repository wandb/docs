---
title: How do I resolve a run initialization timeout error in wandb?
displayed_sidebar: support
tags:
- connectivity
- crashing and hanging runs
---
To resolve a run initialization timeout error, follow these steps:

- **Retry initialization**: Attempt to restart the run.
- **Check network connection**: Confirm a stable internet connection.
- **Update wandb version**: Install the latest version of wandb.
- **Increase timeout settings**: Modify the `WANDB_INIT_TIMEOUT` environment variable:
  ```python
  import os
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **Enable debugging**: Set `WANDB_DEBUG=true` and `WANDB_CORE_DEBUG=true` for detailed logs.
- **Verify configuration**: Check that the API key and project settings are correct.
- **Review logs**: Inspect `debug.log`, `debug-internal.log`, `debug-core.log`, and `output.log` for errors.
