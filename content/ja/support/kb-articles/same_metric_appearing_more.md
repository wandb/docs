---
title: なぜ同じメトリックが複数回表示されるのですか？
menu:
  support:
    identifier: ja-support-kb-articles-same_metric_appearing_more
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

同じキーでさまざまなデータ型をログする場合、データベース上で分割されます。その結果、UI のドロップダウンに同じメトリック名が複数表示されることになります。まとめられるデータ型は、`number`、`string`、`bool`、`other`（主に配列）、および `wandb` のデータ型（`Histogram` や `Image` など）です。こうした問題を防ぐため、1 つのキーには 1 種類のデータ型だけを送信してください。

メトリック名は大文字と小文字を区別しません。たとえば `"My-Metric"` と `"my-metric"` のように、大小の違いだけで名前を使い分けることは避けてください。