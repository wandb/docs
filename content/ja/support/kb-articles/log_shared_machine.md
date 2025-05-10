---
title: 共有マシンで正しい wandb User にログするにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_shared_machine
support:
  - logs
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: log_shared_machine
---
共有マシンを使用する場合、`WANDB_API_KEY` 環境変数を設定して認証を行うことで、正しい WandB アカウントに run がログを生成するようにしてください。環境でソースされている場合、この変数はログイン時に正しい認証情報を提供します。あるいは、スクリプト内で直接環境変数を設定することもできます。

`export WANDB_API_KEY=X` コマンドを実行し、X をあなたの API キーに置き換えてください。ログインしているユーザーは、自分の API キーを [wandb.ai/authorize](https://app.wandb.ai/authorize) で見つけることができます。