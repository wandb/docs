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

W&B エンティティに対して run をログに記録する際の権限エラーを解決するには、次の手順に従います。

- **エンティティとプロジェクト名の確認**: コード内の W&B エンティティとプロジェクト名のスペルと大文字小文字の区別が正しいことを確認してください。
- **権限の確認**: 管理者によって必要な権限が付与されていることを確認してください。
- **ログイン資格情報の確認**: 正しい W&B アカウントにログインしていることを確認してください。次のコードを使用して run を作成し、テストします。
  
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```

- **APIキーの設定**: `WANDB_API_KEY` 環境変数を使用します。
  
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```

- **ホスト情報の確認**: カスタムデプロイメントにはホストURLを設定します。
  
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```