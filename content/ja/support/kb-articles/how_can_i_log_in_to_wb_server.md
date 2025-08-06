---
title: W&B Server にログインするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ユーザー管理
---

以下のいずれかの方法でログインURLを設定します。

- [環境変数]({{< relref "guides/models/track/environment-variables.md" >}}) `WANDB_BASE_URL` を Server のURLに設定します。
- [`wandb login`]({{< relref "/ref/cli/wandb-login.md" >}}) の `--host` フラグに Server のURLを指定します。