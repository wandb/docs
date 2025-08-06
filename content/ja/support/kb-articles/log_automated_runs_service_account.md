---
title: CI（継続的インテグレーション）や社内ツールからローンチされた run をどのようにログすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
- ログ
---

自動テストや W&B へのログを行う内部ツールを実行するには、チーム設定ページで **Service Account** を作成してください。この操作により、継続的インテグレーションなどの自動化ジョブで使用できるサービス APIキー を利用できるようになります。サービスアカウントのジョブを特定のユーザーに紐付けたい場合は、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境変数 を設定してください。

{{< img src="/images/track/common_questions_automate_runs.png" alt="Service account の作成" >}}