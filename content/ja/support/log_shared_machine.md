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

共有マシンを使用する場合、認証のために `WANDB_API_KEY` 環境変数を設定して、run が正しい WandB アカウントに ログ 記録されるようにしてください。環境内でソースされる場合、この変数はログイン時に正しい認証情報を提供します。または、スクリプトで環境変数を直接設定します。

`export WANDB_API_KEY=X` コマンド を実行し、X を API キー に置き換えます。ログイン している ユーザー は、[wandb.ai/authorize](https://app.wandb.ai/authorize) で API キー を確認できます。
