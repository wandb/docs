---
title: How do I resolve permission errors when logging a run to a WandB entity?
displayed_sidebar: support
tags:
- runs
- administrator
---
To resolve permission errors when logging a run to a WandB entity, follow these steps:

- **Verify Entity and Project Names**: Ensure the correct spelling and case sensitivity of the WandB entity and project names.
- **Confirm Permissions**: Make sure you have the necessary permissions from the project owner.
- **Check Log-In Credentials**: Confirm you are logged into the correct WandB account. Test by creating a test run:
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **Set API Key**: Configure your WandB API key:
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **Confirm Host Information**: For custom deployments, set the host correctly:
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```

If issues persist, contact WandB support for further assistance.