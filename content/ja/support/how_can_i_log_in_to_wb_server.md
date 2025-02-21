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

ログイン URL を設定するには、次のいずれかのメソッドを使用します:

- [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_BASE_URL` を サーバー URL に設定します。
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) の `--host` フラグを サーバー URL に設定します。