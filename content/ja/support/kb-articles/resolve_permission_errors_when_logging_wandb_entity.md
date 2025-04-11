---
title: run の権限エラーをログで解決するにはどうすればいいですか?
menu:
  support:
    identifier: ja-support-kb-articles-resolve_permission_errors_when_logging_wandb_entity
support:
- run
- セキュリティ
toc_hide: true
type: docs
url: /support/:filename
---

W&B エンティティへの run ログ中に権限エラーが発生した場合は、次の手順を実行します:

- **エンティティとプロジェクト名の確認**: コード内の W&B エンティティとプロジェクト名のスペルと大文字小文字を確認します。
- **権限の確認**: 管理者によって必要な権限が付与されていることを確認します。
- **ログイン資格情報の確認**: 正しい W&B アカウントにログインしていることを確認します。次のコードを使用して run を作成してテストします:
  
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
  
- **API キーの設定**: `WANDB_API_KEY` 環境変数を使用します:
  
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
  
- **ホスト情報の確認**: カスタムデプロイメントの場合、ホスト URL を設定します:
  
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```