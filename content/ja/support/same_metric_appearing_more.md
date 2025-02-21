---
title: Why is the same metric appearing more than once?
menu:
  support:
    identifier: ja-support-same_metric_appearing_more
tags:
- experiments
toc_hide: true
type: docs
---

同じ キー でさまざまなデータ型を ログ に記録する場合、データベース内でそれらを分割します。これにより、UI ドロップダウンに同じ指標名のエントリが複数作成されます。グループ化されるデータの型は、`number`、`string`、`bool`、`other`（主に配列）、および `Histogram` や `Image` などの `wandb` データ型です。この問題を回避するには、 キー ごとに 1 つの型のみを送信してください。

指標名は大文字と小文字が区別されません。`"My-Metric"` や `"my-metric"` など、大文字と小文字のみが異なる名前の使用は避けてください。
