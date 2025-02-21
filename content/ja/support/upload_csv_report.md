---
title: Upload a CSV to a report
menu:
  support:
    identifier: ja-support-upload_csv_report
tags:
- reports
toc_hide: true
type: docs
---

レポートに CSV をアップロードするには、`wandb.Table` フォーマットを使用します。Python スクリプトで CSV を読み込み、それを `wandb.Table` オブジェクトとしてログに記録します。この操作により、データがレポート内でテーブルとして表示されます。
