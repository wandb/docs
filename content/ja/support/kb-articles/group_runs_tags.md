---
title: タグで Runs をグループ化できますか？
menu:
  support:
    identifier: ja-support-kb-articles-group_runs_tags
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

1 つの run は複数のタグを持てるため、タグによるグループ化はサポートされていません。代わりに、これらの run には [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに値を追加し、この config の値でグループ化してください。これは [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使って実現できます。