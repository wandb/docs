---
title: How do I resolve permission errors when logging a run?
toc_hide: true
type: docs
tags:
- runs
- security
---
To resolve permission errors when logging a run to a W&B entity, follow these steps:

- **Verify entity and project names**: Ensure correct spelling and case sensitivity of the W&B entity and project names in your code.
- **Confirm permissions**: Ensure necessary permissions have been granted by the administrator.
- **Check log-in credentials**: Confirm log-in to the correct W&B account. Test by creating a run with the following code:
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **Set API key**: Use the `WANDB_API_KEY` environment variable:
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **Confirm host information**: For custom deployments, set the host URL:
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```