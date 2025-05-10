---
url: /support/:filename
title: Can you group runs by tags?
toc_hide: true
type: docs
support:
- runs
translationKey: group_runs_tags
---
A run can have multiple tags, so grouping by tags is not supported. Add a value to the [`config`]({{< relref "/guides/models/track/config.md" >}}) object for these runs and group by this config value instead. This can be accomplished using [the API]({{< relref "/guides/models/track/config.md#set-the-configuration-after-your-run-has-finished" >}}).