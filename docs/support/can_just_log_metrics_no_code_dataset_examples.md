---
title: "Can I just log metrics, no code or dataset examples?"
displayed_sidebar: support
tags:
   - None
---
**Dataset Examples**

By default, the platform does not log dataset examples. Enable this feature to view example predictions in the web interface.

**Code Logging**

Two methods exist to turn off code logging:

1. Set `WANDB_DISABLE_CODE` to `true` to turn off all code tracking. This action prevents retrieval of the git SHA and the diff patch.
2. Set `WANDB_IGNORE_GLOBS` to `*.patch` to stop syncing the diff patch to the servers, while keeping it available locally for application with `wandb restore`.