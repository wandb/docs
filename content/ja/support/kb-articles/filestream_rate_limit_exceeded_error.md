---
title: Filestream のレート制限超過エラーを解決するにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 接続
- 障害
---

W&B で「Filestream rate limit exceeded」エラーが発生した場合は、以下の手順で解決してください。

**ログの最適化**:
  - ログの頻度を減らす、またはログをまとめて送信し、API リクエストの回数を減らします。
  - 実験の開始タイミングをずらし、API リクエストが同時に集中しないようにします。

**障害の確認**:
  - 一時的なサーバー側の問題ではないか、[W&B ステータス更新](https://status.wandb.com)で確認します。

**サポートへの問い合わせ**:
  - 実験環境の詳細とともに W&B サポート（support@wandb.com）へ連絡し、レートリミットの増加を依頼してください。