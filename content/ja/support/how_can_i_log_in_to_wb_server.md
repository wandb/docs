---
title: How can I log in to W&B Server?
menu:
  support:
    identifier: ja-support-how_can_i_log_in_to_wb_server
tags:
- user management
toc_hide: true
type: docs
---

ログインURLは、以下のいずれかのメソッドで設定します。

- [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_BASE_URL` に Server URL を設定します。
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) の `--host` フラグに Server URL を設定します。
