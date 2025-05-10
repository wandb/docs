---
title: ノートブックで "Run Finished" アラートは機能しますか?
menu:
  support:
    identifier: ja-support-kb-articles-run_finished_alerts
support:
  - alerts
  - notebooks
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: run_finished_alerts
---
No. **Run Finished** アラート (**Run Finished** 設定をユーザー設定で有効にする) は Python スクリプトでのみ機能し、各セルの実行に対する通知を避けるために Jupyter ノートブック 環境では無効になっています。

ノートブック 環境では `wandb.alert()` を使用してください。