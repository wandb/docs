---
title: W&B サーバーにログインするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_log_in_to_wb_server
support:
- ユーザー管理
toc_hide: true
type: docs
url: /support/:filename
---

以下のいずれかの方法でログイン URL を設定します。

- [環境変数]({{< relref path="guides/models/track/environment-variables.md" lang="ja" >}}) `WANDB_BASE_URL` を Server URL に設定します。
- [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) の `--host` フラグに Server URL を指定します。