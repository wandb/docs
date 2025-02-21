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

run には複数のタグを付けることができるため、タグでグループ化することはサポートされていません。これらの run のために[`設定`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに 値 を追加し、この設定値でグループ化してください。これは [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使用して実現できます。