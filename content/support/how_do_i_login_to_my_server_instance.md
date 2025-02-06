---
title: How can I log in to W&B Server?
toc_hide: true
type: docs
tags:
  - user management
---

Set the login URL by either of these methods:

- Set the [environment variable]({{< relref "/guides/track/environment-variables.md" >}}) `WANDB_BASE_URL` to the Server URL.
- Set the `--host` flag of [`wandb login`]({{< relref "/ref/cli/wandb-login.md" >}}) to the Server URL.

- Setting the environment variable `WANDB_BASE_URL=<instance-url>`
- Adding `--host=<instance-url>` to `wandb login --relogin --host=<instance-url>`
