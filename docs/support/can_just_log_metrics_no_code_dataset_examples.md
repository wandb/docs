---
title: "Can I just log metrics, no code or dataset examples?"
tags:
   - None
---
**Dataset Examples**

By default, the platform does not log dataset examples. Enable this feature to view example predictions in the web interface.

**Code Logging**

Two methods exist to disable code logging:

1. Set `WANDB_DISABLE_CODE` to `true` to disable all code tracking. This action prevents retrieval of the git SHA and the diff patch.
2. Set `WANDB_IGNORE_GLOBS` to `*.patch` to stop syncing the diff patch to the servers, while keeping it available locally for application with `wandb restore`.