---
title: "Can I just log metrics, no code or dataset examples?"
tags:
   - administrator
   - team management
---

By default, W&B does not log dataset examples. By default, W&B logs code and system metrics.

Two methods exist to turn off code logging with environment variables:

1. Set `WANDB_DISABLE_CODE` to `true` to turn off all code tracking. This action prevents retrieval of the git SHA and the diff patch.
2. Set `WANDB_IGNORE_GLOBS` to `*.patch` to stop syncing the diff patch to the servers, while keeping it available locally for application with `wandb restore`.

As an administrator, you can also turn off code saving for your team in your team's settings:

1. Navigate to the settings of your team at `https://wandb.ai/<team>/settings`. Where `<team>` is the name of your team.
2. Scroll to the Privacy section.
3. Toggle **Enable code saving by default**. 