---
title: How can I log in to W&B Server?
toc_hide: true
type: docs
tags:
  - user management
---

To login to a server instance, you should point to its URL by either:
- Setting the environment variable `WANDB_BASE_URL=<instance-url>`
- Adding `--host=<instance-url>` to `wandb login --relogin --host=<instance-url>`
