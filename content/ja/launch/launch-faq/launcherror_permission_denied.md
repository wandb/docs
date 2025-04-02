---
title: How do I fix a "permission denied" error in Launch?
menu:
  launch:
    identifier: ja-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` というエラーメッセージが表示された場合、これは目的の プロジェクト にログを記録するための権限が不十分であることを示しています。考えられる原因は次のとおりです。

1. このマシンにログインしていません。 コマンドライン で[`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。
2. 指定された Entity が存在しません。 Entity は、 ユーザー 名または既存の Team 名である必要があります。必要に応じて、[Subscriptions page](https://app.wandb.ai/billing) で Team を作成してください。
3. プロジェクト の権限がありません。 プロジェクト の作成者にプライバシー設定を **Open** に変更して、 run を プロジェクト に記録できるように依頼してください。