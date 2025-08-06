---
title: CSV をレポートにアップロードする
url: /support/:filename
toc_hide: true
type: docs
support:
- レポート
---

レポートに CSV をアップロードするには、`wandb.Table` フォーマットを使用します。Python スクリプトで CSV を読み込み、`wandb.Table` オブジェクトとしてログしてください。この操作で、データがレポート内のテーブルとして表示されます。