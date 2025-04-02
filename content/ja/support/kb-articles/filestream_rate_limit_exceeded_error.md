---
title: How can I resolve the Filestream rate limit exceeded error?
menu:
  support:
    identifier: ja-support-kb-articles-filestream_rate_limit_exceeded_error
support:
- connectivity
- outage
toc_hide: true
type: docs
url: /support/:filename
---

Weights & Biases (W&B) で「Filestream レート制限超過」エラーを解決するには、次の手順に従ってください。

**ログ記録の最適化**:
  - ログ記録の頻度を減らすか、ログをバッチ処理して、API リクエストを減らします。
  - 実験の開始時間をずらして、同時 API リクエストを回避します。

**停止の確認**:
  - [W&B のステータスアップデート](https://status.wandb.com) を確認して、一時的なサーバー側の問題が原因ではないことを確認します。

**サポートへのお問い合わせ**:
  - 実験設定の詳細を添えて W&B サポート (support@wandb.com) に連絡し、レート制限の引き上げをリクエストします。
