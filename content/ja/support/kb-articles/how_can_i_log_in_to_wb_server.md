---
title: W&B サーバーにログインするには？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_log_in_to_wb_server
support:
- ユーザー管理
toc_hide: true
type: docs
url: /support/:filename
---

ログイン URL を次のいずれかの方法で設定します:

- [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_BASE_URL` をサーバーの URL に設定します。
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) の `--host` フラグをサーバーの URL に設定します。