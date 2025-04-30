---
title: レポートに CSV をアップロードする
menu:
  support:
    identifier: ja-support-kb-articles-upload_csv_report
support:
- レポート
toc_hide: true
type: docs
url: /ja/support/:filename
---

レポートに CSV をアップロードするには、`wandb.Table` フォーマットを使用します。Python スクリプトで CSV を読み込み、`wandb.Table` オブジェクトとしてログします。この操作により、データがレポート内でテーブルとして表示されます。