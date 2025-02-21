---
title: How do I log to the right wandb user on a shared machine?
menu:
  support:
    identifier: ja-support-log_shared_machine
tags:
- logs
toc_hide: true
type: docs
---

共有マシンを使用する場合、`WANDB_API_KEY` 環境変数を設定して認証することで、正しい WandB アカウントにrunをログすることを確認してください。環境で指定されている場合、この変数はログイン時に正しい資格情報を提供します。あるいは、スクリプト内で直接環境変数を設定することもできます。

コマンド `export WANDB_API_KEY=X` を実行し、X をあなたのAPIキーに置き換えてください。ログインしているユーザーは、自分のAPIキーを [wandb.ai/authorize](https://app.wandb.ai/authorize) で見つけることができます。