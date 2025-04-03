---
title: Do "Run Finished" alerts work in notebooks?
menu:
  support:
    identifier: ja-support-kb-articles-run_finished_alerts
support:
- alerts
- notebooks
toc_hide: true
type: docs
url: /support/:filename
---

いいえ。 **Run Finished** アラート（ユーザー 設定の **Run Finished** 設定で有効化）は、Python スクリプトでのみ動作し、Jupyter Notebook 環境では、各セルの実行に対する通知を避けるためにオフのままになっています。

代わりに、ノートブック 環境で `wandb.alert()` を使用してください。
