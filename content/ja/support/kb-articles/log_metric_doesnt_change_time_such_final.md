---
title: 最終的な評価精度のように時間経過で変化しないメトリクスは、どのようにログすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metric_doesnt_change_time_such_final
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

`run.log({'final_accuracy': 0.9})` を使うと、最終精度が正しく更新されます。デフォルトでは、`run.log({'final_accuracy': <value>})` は `run.settings['final_accuracy']` を更新し、これは Runs テーブルの 値 を反映します。