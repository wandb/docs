---
title: 共有マシンで正しい wandb ユーザーにログするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- ログ
---

共有マシンを使用する場合は、`WANDB_API_KEY` 環境変数を設定して、Runs が正しい WandB アカウントにログされるように認証を行ってください。環境にこの変数が設定されていると、ログイン時に正しい認証情報が提供されます。または、環境変数をスクリプト内で直接設定することもできます。

コマンド `export WANDB_API_KEY=X` を実行し、X を自分の APIキー に置き換えてください。ログインしているユーザーは、自分の APIキー を [wandb.ai/authorize](https://app.wandb.ai/authorize) で確認できます。