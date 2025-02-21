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

同じキーの下に様々なデータ型をログする際、データベースでそれらを分けてください。これにより、UIのドロップダウンに同じメトリック名が複数のエントリとして表示されます。グループ化されるデータ型には `number`、`string`、`bool`、`other`（主に配列）、および `Histogram` や `Image` などの `wandb` データ型があります。この問題を防ぐために、1 つのキーに対して 1 つのデータ型のみを送信してください。

メトリック名は大文字小文字を区別しません。`"My-Metric"` と `"my-metric"` のように、大文字小文字のみが異なる名前の使用は避けてください。