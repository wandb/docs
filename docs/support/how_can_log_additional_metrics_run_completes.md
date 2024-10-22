---
title: "How can I log additional metrics after a run completes?"
tags:
   - None
---

There are several ways to do this.

For complicated workflows, we recommend using multiple runs and setting group parameters in [`wandb.init`](../guides/track/launch.md) to a unique value in all the processes that are run as part of a single experiment. The [runs table](../guides/app/pages/run-page.md) will automatically group the table by the group ID and the visualizations will behave as expected. This will allow you to run multiple experiments and training runs as separate processes log all the results into a single place.

For simpler workflows, you can call `wandb.init` with `resume=True` and `id=UNIQUE_ID` and then later call `wandb.init` with the same `id=UNIQUE_ID`. Then you can log normally with [`wandb.log`](../guides/track/log/intro.md) or `wandb.summary` and the runs values will update.


## Performance