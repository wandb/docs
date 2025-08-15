---
title: 継続的インテグレーションや社内ツールでローンチされた run をログするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_automated_runs_service_account
support:
- run
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

自動テストや W&B へのログを行う内部ツールを起動するには、チームの設定ページで **Service Account** を作成してください。この操作により、継続的インテグレーションなどで動作する自動ジョブ用にサービス用の APIキー を利用できるようになります。Service Account のジョブを特定のユーザーに紐付けるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` の環境変数を設定してください。

{{< img src="/images/track/common_questions_automate_runs.png" alt="Service Account の作成" >}}