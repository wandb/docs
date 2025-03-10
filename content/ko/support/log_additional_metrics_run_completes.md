---
menu:
  support:
    identifier: ko-support-log_additional_metrics_run_completes
tags:
- runs
- metrics
title: How can I log additional metrics after a run completes?
toc_hide: true
type: docs
---

There are several ways to manage experiments.

For complex workflows, use multiple runs and set the group parameters in [`wandb.init`]({{< relref path="/guides/models/track/launch.md" lang="ko" >}}) to a unique value for all processes within a single experiment. The [**Runs** tab]({{< relref path="/guides/models/track/project-page.md#runs-tab" lang="ko" >}}) will group the table by group ID, ensuring that visualizations function properly. This approach enables concurrent experiments and training runs while logging results in one location.

For simpler workflows, call `wandb.init` with `resume=True` and `id=UNIQUE_ID`, then call `wandb.init` again with the same `id=UNIQUE_ID`. Log normally with [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ko" >}}) or `wandb.summary`, and the run values will update accordingly.