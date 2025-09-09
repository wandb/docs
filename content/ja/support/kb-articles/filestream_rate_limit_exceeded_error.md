---
title: Filestream rate limit exceeded エラーを解決するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-filestream_rate_limit_exceeded_error
support:
- 接続性
- 障害
toc_hide: true
type: docs
url: /support/:filename
---

W&B で「Filestream rate limit exceeded」エラーを解決するには、次の手順に従ってください:

**ログを最適化**:
  - ログの頻度を下げる、または ログ をバッチ化して API リクエストを減らす。
  - 同時の API リクエストを避けるため、実験の開始時刻をずらす。

**障害の有無を確認**:
  - 一時的なサーバー側の問題が原因でないか、[W&B ステータス更新](https://status.wandb.com) を確認する。

**サポートに連絡**:
  - レート制限の引き上げを依頼するため、実験のセットアップの詳細を添えて W&B サポート (support@wandb.com) に連絡する。