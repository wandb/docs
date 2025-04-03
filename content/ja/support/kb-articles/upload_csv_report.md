---
title: Upload a CSV to a report
menu:
  support:
    identifier: ja-support-kb-articles-upload_csv_report
support:
- reports
toc_hide: true
type: docs
url: /support/:filename
---

CSV を Report にアップロードするには、`wandb.Table` 形式を使用します。Python スクリプトで CSV をロードし、`wandb.Table` オブジェクトとしてログに記録します。この操作により、データが Report 内のテーブルとしてレンダリングされます。
