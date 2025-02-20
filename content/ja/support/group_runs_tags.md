---
menu:
  support:
    identifier: ja-support-group_runs_tags
tags:
- runs
title: Can you group runs by tags?
toc_hide: true
type: docs
---

A run can have multiple tags, so grouping by tags is not supported. Add a value to the [`config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) object for these runs and group by this config value instead. This can be accomplished using [the API]({{< relref path="/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" lang="ja" >}}).