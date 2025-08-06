---
menu:
  support:
    identifier: ko-support-kb-articles-logging_turn_off
support:
- logs
title: How do I turn off logging?
toc_hide: true
type: docs
url: /support/:filename
---

The command `wandb offline` sets the environment variable `WANDB_MODE=offline`, preventing data from syncing to the remote W&B server. This action affects all projects, stopping the logging of data to W&B servers.

To suppress warning messages, use the following code:

```python
import logging

logger = logging.getLogger("wandb")
logger.setLevel(logging.WARNING)
```