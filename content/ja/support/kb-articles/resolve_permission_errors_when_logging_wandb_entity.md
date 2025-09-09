---
title: run をログする際の権限エラーはどのように解決できますか？
menu:
  support:
    identifier: ja-support-kb-articles-resolve_permission_errors_when_logging_wandb_entity
support:
- runs
- セキュリティ
toc_hide: true
type: docs
url: /support/:filename
---

W&B の Entities に run を記録するときの権限エラーを解決するには、次の手順に従ってください:

- **Entities と Projects の名前を確認**: コード内の W&B の Entities と Projects の名前の綴りと大文字・小文字が正しいことを確認します。
- **権限を確認**: 管理者から必要な権限が付与されていることを確認します。
- **ログイン資格情報を確認**: 正しい W&B アカウントにログインしていることを確認します。次のコードで run を作成してテストします:
  ```python
  import wandb

  run = wandb.init(entity="your_entity", project="your_project")
  run.log({'example_metric': 1})
  run.finish()
  ```
- **APIキーを設定**: `WANDB_API_KEY` 環境変数を使用します:
  ```bash
  export WANDB_API_KEY='your_api_key'
  ```
- **ホスト情報を確認**: カスタム デプロイメントの場合は、ホスト URL を設定します:
  ```bash
  wandb login --relogin --host=<host-url>
  export WANDB_BASE_URL=<host-url>
  ```