---
title: 継続的インテグレーションや社内ツールによって起動された runs をログに記録するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_automated_runs_service_account
support:
- runs
- ログ
toc_hide: true
type: docs
url: /support/:filename
---

W&B にログを記録する自動テストや社内ツールを実行するには、チームの設定ページで **Service Account** を作成してください。これにより、継続的インテグレーション (CI) を含む自動ジョブで使用するためのサービス用 APIキーを利用できます。Service Account のジョブを特定のユーザーに紐づけるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境変数を設定してください。

{{< img src="/images/track/common_questions_automate_runs.png" alt="Service Account の作成" >}}