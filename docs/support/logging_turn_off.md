---
title: How do I turn off logging?
displayed_sidebar: support
tags:
- logs
---
The command `wandb offline` sets an environment variable, `WANDB_MODE=offline` . This stops any data from syncing from your machine to the remote wandb server. If you have multiple projects, they will all stop syncing logged data to W&B servers.

To quiet the warning messages:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```