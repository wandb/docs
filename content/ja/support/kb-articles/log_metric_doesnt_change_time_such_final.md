---
title: 時間とともに変化しないメトリクス、例えば最終的な評価精度をログするにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

`run.log({'final_accuracy': 0.9})` を使うと、最終的な accuracy が正しく更新されます。デフォルトでは、`run.log({'final_accuracy': <value>})` は `run.settings['final_accuracy']` を更新し、この値は Runs テーブルに反映されます。