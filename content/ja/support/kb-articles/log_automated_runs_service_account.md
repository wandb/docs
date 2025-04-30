---
title: 継続的なインテグレーションや内部ツールによってローンンチされた run をどのようにログしますか？
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

W&B にログを記録する自動テストや内部ツールをローンチするには、チーム設定ページで **Service Account** を作成します。このアクションにより、継続的インテグレーションを通じて実行されるものを含む自動化されたジョブのためのサービス API キーの使用が可能になります。サービスアカウントジョブを特定のユーザーに帰属させるには、`WANDB_USERNAME` または `WANDB_USER_EMAIL` 環境変数を設定します。

{{< img src="/images/track/common_questions_automate_runs.png" alt="自動化されたジョブのためにチーム設定ページでサービスアカウントを作成する" >}}