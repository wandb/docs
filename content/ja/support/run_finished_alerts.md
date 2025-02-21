---
title: Do "Run Finished" alerts work in notebooks?
menu:
  support:
    identifier: ja-support-run_finished_alerts
tags:
- alerts
- notebooks
toc_hide: true
type: docs
---

いいえ。**Run Finished** アラート（ ユーザー 設定 の **Run Finished** 設定 で有効化）は、Python スクリプト でのみ動作し、Jupyter ノートブック 環境 では、各セルの実行に関する通知を避けるためにオフのままになります。

ノートブック 環境 で `wandb.alert()` を代わりに使用してください。
