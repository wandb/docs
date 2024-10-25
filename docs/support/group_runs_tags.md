---
title: Can you group runs by tags?
tags:
- tags
- runs
---
Because a run can have multiple tags, we don't support grouping by them. Our recommendation would be to add a value to the [`config`](../guides/track/config.md) object of these runs and then group by this config value. You can do this with [our API](../guides/track/config#set-the-configuration-after-your-run-has-finished).