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

**Run Finished** アラート （ユーザー設定の **Run Finished** 設定で有効にする）は Python スクリプトでのみ動作し、各セルの実行について通知が行われないよう Jupyter Notebook 環境ではオフのままです。

代わりに、ノートブック環境では `wandb.alert()` を使用してください。