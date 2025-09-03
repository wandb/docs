---
title: Launch で "permission denied" エラーを修正するにはどうすればよいですか？
menu:
  launch:
    identifier: ja-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

エラーメッセージ `Launch Error: Permission denied` に遭遇した場合、これは、目的のプロジェクトにログを記録するための権限が不十分であることを示しています。考えられる原因は次のとおりです：

1. このマシンにログインしていません。コマンドラインで [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。
2. 指定されたエンティティが存在しません。エンティティは、ユーザーのユーザー名または既存のチームの名前である必要があります。必要に応じて、[Subscriptions page](https://app.wandb.ai/billing) でチームを作成してください。
3. プロジェクトの権限がありません。プロジェクトの作成者にプライバシー設定を **Open** に変更するよう依頼して、プロジェクトに run をログできるようにしてください。