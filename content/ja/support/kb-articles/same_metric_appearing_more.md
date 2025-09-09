---
title: 同じメトリクスが複数回表示されるのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-same_metric_appearing_more
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

同じキーに異なるデータ型をログすると、データベースで分割されます。その結果、UI ドロップダウンに同じメトリクス名が複数の項目として表示されます。グループ化されるデータ型は `number`、`string`、`bool`、`other`（主に配列）、および `Histogram` や `Image` などの任意の `wandb` データ型です。この問題を防ぐには、1 つのキーにつき 1 種類の型だけを送信してください。

メトリクス名は大文字と小文字を区別しません。`"My-Metric"` と `"my-metric"` のように、大文字・小文字の違いだけの名前は避けてください。