---
title: run をタグでグループ化できますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
---

run には複数のタグを設定できますが、タグでのグループ化はサポートされていません。これらの run には [`config`]({{< relref "/guides/models/track/config.md" >}}) オブジェクトに値を追加し、この config の値でグループ化してください。この処理は [API]({{< relref "/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" >}}) を使って実現できます。