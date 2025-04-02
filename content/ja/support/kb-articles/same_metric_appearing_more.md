---
title: Why is the same metric appearing more than once?
menu:
  support:
    identifier: ja-support-kb-articles-same_metric_appearing_more
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

同じキーでさまざまな データ型 を ログ に記録する際に、データベース内でそれらを分割します。これにより、UIドロップダウンに同じ メトリック 名のエントリが複数表示されます。グループ化される データ型 は、`number`、`string`、`bool`、`other`（主に配列）、および`Histogram`や`Image`などの`wandb`の データ型 です。この問題を回避するには、キーごとに1つの型のみを送信してください。

メトリック 名は大文字と小文字が区別されません。`"My-Metric"`や`"my-metric"`のように、大文字と小文字のみが異なる名前の使用は避けてください。
