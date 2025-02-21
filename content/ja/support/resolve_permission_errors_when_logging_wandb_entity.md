---
title: How do I resolve permission errors when logging a run?
menu:
  support:
    identifier: ja-support-resolve_permission_errors_when_logging_wandb_entity
tags:
- runs
- security
toc_hide: true
type: docs
---

W&B entity に run を ログ する際の権限エラーを解決するには、次の手順に従ってください。

- **entity と project 名の確認**: コード 内で、W&B entity と project 名のスペルと大文字と小文字が正しいことを確認してください。
- **権限の確認**: 必要な権限が管理者によって付与されていることを確認してください。
- **ログイン認証情報の確認**: 正しい W&B アカウント にログインしていることを確認してください。次の コード で run を作成してテストします。
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **APIキー の設定**: `WANDB_API_KEY` 環境変数を使用します。
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **ホスト 情報の確認**: カスタム デプロイメント の場合、ホスト URL を設定します。
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```
