---
title: 共有マシンで正しい wandb ユーザーにログするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_shared_machine
support:
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

共有マシンを使用する場合は、`WANDB_API_KEY` 環境変数を設定して認証し、run が正しい WandB アカウントにログされるようにしてください。環境にこの変数を設定しておくと、ログイン時に正しい認証情報が提供されます。あるいは、スクリプト内で直接環境変数を設定することも可能です。

`export WANDB_API_KEY=X` というコマンドを実行し、X を自身の APIキー に置き換えてください。ログイン済みのユーザーは [wandb.ai/authorize](https://app.wandb.ai/authorize) で自身の APIキー を確認できます。