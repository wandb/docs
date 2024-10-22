---
title: "How can I log additional metrics after a run completes?"
tags:
   - None
---
There are several ways to manage experiments.

For complex workflows, use multiple runs and set the group parameters in [`wandb.init`](../guides/track/launch.md) to a unique value for all processes within a single experiment. The [runs table](../guides/app/pages/run-page.md) will group the table by group ID, ensuring that visualizations function properly. This approach enables concurrent experiments and training runs while logging results in one location.

For simpler workflows, call `wandb.init` with `resume=True` and `id=UNIQUE_ID`, then call `wandb.init` again with the same `id=UNIQUE_ID`. Log normally with [`wandb.log`](../guides/track/log/intro.md) or `wandb.summary`, and the run values will update accordingly.

## Performance