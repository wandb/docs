---
title: run をタグでグループ化できますか？
menu:
  support:
    identifier: ja-support-kb-articles-group_runs_tags
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

1 つの run は複数のタグを持つことができますが、タグによるグループ化はサポートされていません。これらの run には [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに値を追加し、この config の値でグループ化してください。これは [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使って実現できます。