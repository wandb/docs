---
title: run をログする際のパーミッションエラーを解決するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
- セキュリティ
---

W&B の Entity に run を記録する際の権限エラーを解決するには、以下の手順を実施してください。

- **Entity および Project 名の確認**: コード内で W&B の Entity 名と Project 名が正しく、かつ大文字・小文字も合っていることを確認してください。
- **権限の確認**: 管理者から必要な権限が付与されているか確認してください。
- **ログイン情報の確認**: 正しい W&B アカウントにログインしているか確認してください。以下のコードで run を作成してテスト可能です。
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **APIキーの設定**: `WANDB_API_KEY` 環境変数を使用してください。
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **ホスト情報の確認**: カスタムデプロイメントの場合はホスト URL を設定してください。
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```