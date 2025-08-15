---
title: Launch で「permission denied」エラーが発生した場合の対処方法
menu:
  launch:
    identifier: ja-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` エラーメッセージが表示された場合は、指定したプロジェクトへのログ権限が不足しています。主な原因は次の通りです。

1. このマシンでログインしていません。コマンドラインで [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。
2. 指定した entity が存在しません。entity には自分のユーザー名、または既存の Team の名前を指定する必要があります。必要に応じて [Subscriptions page](https://app.wandb.ai/billing) から Team を作成してください。
3. プロジェクトへの権限がありません。プロジェクトの作成者に依頼して、プライバシー設定を **Open** に変更してもらうことで、プロジェクトに run をログできるようになります。