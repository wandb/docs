---
title: wandb で run の初期化タイムアウトエラーを解決するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 接続性
- クラッシュやハングする run
---

run の初期化タイムアウトエラーを解決するには、以下の手順をお試しください。

- **初期化の再試行**: run を再起動してみてください。
- **ネットワーク接続の確認**: 安定したインターネット接続を確認してください。
- **wandb のバージョンを更新**: 最新バージョンの wandb をインストールしてください。
- **タイムアウト設定を増やす**: `WANDB_INIT_TIMEOUT` 環境変数を変更します。
  ```python
  import os
  # タイムアウトを 600 秒に設定
  os.environ['WANDB_INIT_TIMEOUT'] = '600'
  ```
- **デバッグを有効にする**: 詳細なログを取得するために、`WANDB_DEBUG=true` および `WANDB_CORE_DEBUG=true` を設定します。
- **設定の確認**: APIキー と project 設定が正しいことを確認してください。
- **ログの確認**: エラーがないか `debug.log`、`debug-internal.log`、`debug-core.log`、`output.log` をチェックしてください。