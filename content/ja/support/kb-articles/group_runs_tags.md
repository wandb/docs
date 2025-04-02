---
title: Can you group runs by tags?
menu:
  support:
    identifier: ja-support-kb-articles-group_runs_tags
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

1 つの run に複数のタグを設定できるため、タグでのグループ化はサポートされていません。代わりに、これらの run の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに 値 を追加し、この config の 値 でグループ化してください。これは、[API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使用して実現できます。
