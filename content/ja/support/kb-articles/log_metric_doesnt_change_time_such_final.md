---
title: 時間とともに変化しない最終的な評価精度などのメトリクスをログするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
---

`run.log({'final_accuracy': 0.9})` を使うと、最終的な accuracy が正しく更新されます。デフォルトでは、`run.log({'final_accuracy': <value>})` は `run.settings['final_accuracy']` を更新し、その値は Runs テーブルにも反映されます。