---
title: '"Run Finished" アラートはノートブックで動作しますか？'
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

いいえ。 **Run Finished** アラート（ユーザー設定の **Run Finished** 設定で有効化）は Python スクリプトでのみ作動し、各セルの実行ごとの通知を避けるため Jupyter ノートブック 環境ではオフのままです。

代わりに ノートブック 環境では `run.alert()` を使用してください。