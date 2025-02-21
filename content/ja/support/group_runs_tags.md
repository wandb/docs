---
title: Can you group runs by tags?
menu:
  support:
    identifier: ja-support-group_runs_tags
tags:
- runs
toc_hide: true
type: docs
---

1 つの run に複数のタグを設定できるため、タグによるグループ化はサポートされていません。これらの run の [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに 値 を追加し、代わりにこの config の 値 でグループ化してください。これは、[API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使用して実現できます。
