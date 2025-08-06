---
title: Launch で「permission denied」エラーが発生した場合の対処方法は？
menu:
  launch:
    identifier: launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` というエラーメッセージが表示された場合、目的のプロジェクトへのログ権限が不足していることを示しています。主な原因には以下が考えられます。

1. このマシンでログインしていません。コマンドラインで [`wandb login`]({{< relref "/ref/cli/wandb-login.md" >}}) を実行してください。
2. 指定した entity が存在しません。entity にはご自身のユーザー名または既存の Team 名が必要です。必要に応じて [Subscriptions ページ](https://app.wandb.ai/billing) から Team を作成してください。
3. プロジェクトへの権限がありません。プロジェクト作成者に依頼してプライバシー設定を **Open** に変更してもらうことで、そのプロジェクトへの run ログが可能になります。