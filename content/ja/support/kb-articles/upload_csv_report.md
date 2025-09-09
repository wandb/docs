---
title: CSV を レポートにアップロードする
menu:
  support:
    identifier: ja-support-kb-articles-upload_csv_report
support:
- レポート
toc_hide: true
type: docs
url: /support/:filename
---

CSV を レポート にアップロードするには、`wandb.Table` フォーマットを使用します。CSV を Python スクリプトで読み込み、`wandb.Table` オブジェクトとしてログします。この操作により、レポート内でデータがテーブルとして表示されます。