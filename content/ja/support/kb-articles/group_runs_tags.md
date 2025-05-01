---
title: ランをタグでグループ化できますか？
menu:
  support:
    identifier: ja-support-kb-articles-group_runs_tags
support:
  - runs
toc_hide: true
type: docs
url: /ja/support/:filename
---
run は複数のタグを持つことができるため、タグによるグループ化はサポートされていません。これらの run に対して [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) オブジェクトに値を追加し、この config 値でグループ化してください。これは [API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}) を使用して達成できます。