---
title: 共有マシンで、正しい wandb user にログを送るにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_shared_machine
support:
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

共有マシンを使用する場合は、認証のために `WANDB_API_KEY` 環境変数を設定し、Runs が正しい WandB アカウントにログされるようにしてください。環境に読み込まれていれば、ログイン時に正しい認証情報が使用されます。あるいは、スクリプト内で環境変数を直接設定します。

コマンド `export WANDB_API_KEY=X` を実行し、X を自分の APIキー に置き換えてください。ログイン済みのユーザーは [wandb.ai/authorize](https://app.wandb.ai/authorize) で APIキー を確認できます。