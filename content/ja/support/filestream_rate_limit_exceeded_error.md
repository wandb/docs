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

"Filestream rate limit exceeded" エラーを Weights & Biases (W&B) で解決するには、以下の手順に従ってください。

**ログの最適化**:
  - ログの頻度を減らすか、ログをバッチ化して API リクエストを減らします。
  - 実験の開始時間をずらして、同時に API リクエストが発生しないようにします。

**障害を確認する**:
  - 一時的なサーバー側の問題から発生しているかどうかを確認するために、[W&B ステータス更新](https://status.wandb.com)をチェックします。

**サポートに連絡する**:
  - レート制限の引き上げを依頼するために、実験のセットアップの詳細を添えて W&B サポート (support@wandb.com) に連絡します。