---
title: How do I resolve permission errors when logging a run?
menu:
  support:
    identifier: ja-support-kb-articles-resolve_permission_errors_when_logging_wandb_entity
support:
- runs
- security
toc_hide: true
type: docs
url: /support/:filename
---

W&B の Entity に run をログする際の権限エラーを解決するには、以下の手順に従ってください。

- **Entity と Project 名の確認**: コード内で、W&B の Entity と Project 名のスペルと大文字小文字が正しいことを確認してください。
- **権限の確認**: 必要な権限が管理者によって付与されていることを確認してください。
- **ログイン認証情報の確認**: 正しい W&B アカウントにログインしていることを確認してください。次のコードで run を作成してテストします。
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **APIキー の設定**: `WANDB_API_KEY` 環境 変数を使用します。
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **ホスト情報の確認**: カスタム デプロイメント の場合、ホスト URL を設定します。
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```
