---
title: Filestream のレート制限超過エラーを解決するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-filestream_rate_limit_exceeded_error
support:
- 接続
- 障害
toc_hide: true
type: docs
url: /support/:filename
---

W&B で「Filestream rate limit exceeded」エラーが発生した場合は、以下の手順に従ってください。

**ログの最適化**:
  - ログの頻度を減らしたり、ログをバッチ処理することで API リクエスト数を削減します。
  - 実験の開始タイミングをずらして、同時に API リクエストが集中しないようにします。

**障害の確認**:
  - 一時的なサーバー側の問題でないか、[W&B ステータス更新](https://status.wandb.com)を確認してください。

**サポートへの連絡**:
  - レートリミットの引き上げをリクエストする場合は、実験構成の詳細を添えて W&B サポート（support@wandb.com）までご連絡ください。