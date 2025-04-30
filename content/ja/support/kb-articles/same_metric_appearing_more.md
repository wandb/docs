---
title: なぜ同じ指標が複数回表示されるのでしょうか？
menu:
  support:
    identifier: ja-support-kb-articles-same_metric_appearing_more
support:
- 実験管理
toc_hide: true
type: docs
url: /ja/support/:filename
---

同じキーの下にさまざまなデータ型をログする場合、それらをデータベースで分割します。これにより、UIのドロップダウンに同じメトリクス名の複数のエントリが表示されます。グループ化されるデータ型は `number`、`string`、`bool`、`other`（主に配列）および `Histogram` や `Image` などの任意の `wandb` データ型です。この問題を防ぐために、キーごとに一種類のみ送信してください。

メトリクス名は大文字と小文字が区別されません。`"My-Metric"` と `"my-metric"` のように、大文字と小文字だけが異なる名前の使用を避けてください。