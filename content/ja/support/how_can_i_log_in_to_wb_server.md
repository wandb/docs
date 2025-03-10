---
menu:
  support:
    identifier: ja-support-how_can_i_log_in_to_wb_server
tags:
- user management
title: How can I log in to W&B Server?
toc_hide: true
type: docs
---

Set the login URL by either of these methods:

- Set the [environment variable]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_BASE_URL` to the Server URL.
- Set the `--host` flag of [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) to the Server URL.