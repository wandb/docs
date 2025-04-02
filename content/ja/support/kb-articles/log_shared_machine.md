---
title: How do I log to the right wandb user on a shared machine?
menu:
  support:
    identifier: ja-support-kb-articles-log_shared_machine
support:
- logs
toc_hide: true
type: docs
url: /support/:filename
---

共有マシンを使用する場合、認証のために `WANDB_API_KEY` 環境変数を設定して、run が正しい WandB アカウントにログ記録されるようにしてください。環境でソースされる場合、この変数はログイン時に正しい認証情報を提供します。または、スクリプトで環境変数を直接設定します。

`export WANDB_API_KEY=X` コマンドを実行し、X を API キーに置き換えます。ログインしている ユーザー は、[wandb.ai/authorize](https://app.wandb.ai/authorize) で API キーを確認できます。
