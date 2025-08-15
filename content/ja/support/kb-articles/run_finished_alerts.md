---
title: ノートブックで「Run Finished」アラートは機能しますか？
menu:
  support:
    identifier: ja-support-kb-articles-run_finished_alerts
support:
- アラート
- ノートブック
toc_hide: true
type: docs
url: /support/:filename
---

いいえ。**Run Finished** アラート（ユーザー設定の **Run Finished** 設定で有効化）は Python スクリプトでのみ動作し、各セルの実行ごとに通知が届くのを防ぐため、Jupyter Notebook 環境では無効のままです。

ノートブック環境では代わりに `run.alert()` をご利用ください。