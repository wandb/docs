---
title: How do I log runs launched by continuous integration or internal tools?
menu:
  support:
    identifier: ja-support-log_automated_runs_service_account
tags:
- runs
- logs
toc_hide: true
type: docs
---

自動テストや W&B に ログ を記録する内部 ツール を ローンチ するには、 チーム 設定 ページで _**サービス アカウント**_ を作成します。この操作により、継続的 インテグレーション を通じて実行される ジョブ を含む、自動化された ジョブ に サービス API キー を使用できます。サービス アカウント の ジョブ を特定の ユーザー に関連付けるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境 変数 を設定します。

{{< img src="/images/track/common_questions_automate_runs.png" alt="自動化されたジョブのためにチーム設定ページでサービスアカウントを作成する" >}}
