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

W&B にログを記録する自動化テストや内部ツールをローンチするには、チーム設定ページで **Service Account** を作成してください。この操作により、継続的インテグレーションを通じて実行されるものを含む自動ジョブ用のサービス API キーを使用できます。特定のユーザーにサービスアカウントジョブを割り当てるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境変数を設定します。

{{< img src="/images/track/common_questions_automate_runs.png" alt="自動化されたジョブのためにチーム設定ページでサービスアカウントを作成する" >}}