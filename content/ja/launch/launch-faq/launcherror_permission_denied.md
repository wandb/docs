---
title: Launch で "permission denied" エラーを解決するには？
menu:
  launch:
    identifier: ja-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

`Launch Error: Permission denied` というエラーメッセージが表示された場合、目的の project にログするための権限が不足しています。考えられる原因は次のとおりです:

1. このマシンでログインしていません。コマンドラインで [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。
2. 指定した entity が存在しません。entity はあなたのユーザー名、または既存の team の名前である必要があります。必要に応じて [Subscriptions ページ](https://app.wandb.ai/billing) から team を作成してください。
3. project の権限がありません。project に run をログできるように、project の作成者にプライバシー設定を **Open** に変更するよう依頼してください。