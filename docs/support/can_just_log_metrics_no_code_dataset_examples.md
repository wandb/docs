---
title: "Can I just log metrics, no code or dataset examples?"
tags:
   - None
---

**Dataset Examples**

By default, we don't log any of your dataset examples. You can explicitly turn this feature on to see example predictions in our web interface.

**Code Logging**

There are two ways to turn off code logging:

1. Set `WANDB_DISABLE_CODE` to `true` to turn off all code tracking. We won't pick up the git SHA or the diff patch.
2. Set `WANDB_IGNORE_GLOBS` to `*.patch` to turn off syncing the diff patch to our servers. You'll still have it locally and be able to apply it with the `wandb restore`.