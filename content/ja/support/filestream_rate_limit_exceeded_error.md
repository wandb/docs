---
title: How can I resolve the Filestream rate limit exceeded error?
menu:
  support:
    identifier: ja-support-filestream_rate_limit_exceeded_error
tags:
- connectivity
- outage
toc_hide: true
type: docs
---

Weights & Biases (W&B) で「Filestream rate limit exceeded」エラーを解決するには、以下の手順に従ってください。

**ログ記録の最適化**:
  - ログ記録の頻度を減らすか、ログをバッチ処理して API リクエストを減らします。
  - 実験 の開始時間をずらして、API リクエストの同時実行を避けます。

**停止の確認**:
  - [W&B ステータスアップデート](https://status.wandb.com) を確認して、一時的な サーバー 側の問題が発生していないことを確認します。

**サポートへのお問い合わせ**:
  - W&B サポート (support@wandb.com) に 実験 の設定の詳細を添えて連絡し、レート制限の引き上げをリクエストしてください。
