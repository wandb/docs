---
title: run をログする際の権限エラーはどう解決すればよいですか？
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

W&B の Entity に run をログする際の権限エラーを解決するには、以下の手順を実行してください。

- **Entity と Project 名の確認**: コード内で W&B の Entity 名と Project 名のスペルや大文字・小文字が正しいか確認してください。
- **権限の確認**: 管理者によって必要な権限が付与されているか確認してください。
- **ログイン情報の確認**: 正しい W&B アカウントにログインしているか確認してください。以下のコードで run を作成し、テストできます。
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})  # メトリクスの例をログ
  run.finish()  # run を終了
  ```
- **APIキーの設定**: `WANDB_API_KEY` 環境変数を利用してください。
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **ホスト情報の確認**: カスタムデプロイメントの場合は、ホスト URL を設定してください。
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```