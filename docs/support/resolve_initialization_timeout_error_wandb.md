---
title: How do I resolve a run initialization timeout error in wandb?
displayed_sidebar: support
tags:
- connectivity
- crashing and hanging runs
---
To resolve a run initialization timeout error in W&B (Weights & Biases), try the following steps:

- **Retry Initialization**: Sometimes, the issue may resolve by retrying the run.
- **Check Network Connection**: Ensure a stable internet connection.
- **Update wandb Version**: Use the latest version of wandb.
- **Increase Timeout Settings**: Adjust`WANDB_INIT_TIMEOUT` environment variable:
  ```python
  import os
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **Enable Debugging**: Set `WANDB_DEBUG=true` and `WANDB_CORE_DEBUG=true` for detailed logs.
- **Verify Configuration**: Ensure your API key and project settings are correct.
- **Review Logs**: Check `debug.log`, `debug-internal.log`,`debug-core.log`, `output.log` for errors.

If issues persist, contact W&B support with log files and configuration details.
    