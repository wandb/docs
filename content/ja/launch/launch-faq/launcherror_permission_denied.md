---
title: How do I fix a "permission denied" error in Launch?
menu:
  launch:
    identifier: ja-launch-launch-faq-launcherror_permission_denied
    parent: launch-faq
---

エラーメッセージ `Launch Error: Permission denied` が表示された場合、これは指定されたプロジェクトにログする権限が不十分であることを示しています。考えられる原因は以下の通りです：

1. このマシンでログインしていない。コマンドラインで [`wandb login`]({{< relref path="/ref/cli/wandb-login.md" lang="ja" >}}) を実行してください。
2. 指定したエンティティが存在しない。エンティティはユーザー名または既存のチーム名でなければなりません。必要に応じて、[サブスクリプションページ](https://app.wandb.ai/billing) でチームを作成してください。
3. プロジェクトの権限が不足している。プロジェクトの作成者に連絡し、プライバシー設定を **Open** に変更して run をプロジェクトにログできるように依頼してください。