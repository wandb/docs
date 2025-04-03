---
title: How do I log runs launched by continuous integration or internal tools?
menu:
  support:
    identifier: ja-support-kb-articles-log_automated_runs_service_account
support:
- runs
- logs
toc_hide: true
type: docs
url: /support/:filename
---

W&B に ログを記録する自動テストや内部 ツール を Launch するには、 チーム の 設定 ページで**サービス アカウント**を作成します。これにより、継続的 インテグレーション を通じて実行される ジョブ を含む、自動化された ジョブ にサービス API キー を使用できます。サービス アカウント の ジョブ を特定の ユーザー に関連付けるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境 変数 を設定します。

{{< img src="/images/track/common_questions_automate_runs.png" alt="Create a service account on your team settings page for automated jobs" >}}
