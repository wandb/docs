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

CSV を report にアップロードするには、`wandb.Table` 形式を使用します。Python スクリプト で CSV を読み込み、`wandb.Table` オブジェクト として ログ します。この操作により、データ が report にテーブル として表示されます。
