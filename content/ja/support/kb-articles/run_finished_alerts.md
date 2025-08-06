---
title: ノートブックで「Run Finished」アラートは機能しますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アラート
- ノートブック
---

いいえ。**Run Finished** アラート（ユーザー設定内の **Run Finished** 設定で有効化）は Python スクリプトでのみ動作し、Jupyter ノートブック環境では各セル実行ごとに通知が来るのを防ぐため自動的にオフになります。

ノートブック環境では代わりに `run.alert()` を使用してください。