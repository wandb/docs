---
menu:
  support:
    identifier: ko-support-kb-articles-log_list_values
support:
- logs
- experiments
title: How do I log a list of values?
toc_hide: true
type: docs
url: /support/:filename
---

These examples show logging losses a couple of different ways using [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run/#method-runlog/" lang="ko" >}}).

{{< tabpane text=true >}}
{{% tab "Using a dictionary" %}}
```python
import wandb

# Initialize a new run
with wandb.init(project="log-list-values", name="log-dict") as run:
    # Log losses as a dictionary
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": losses})
    run.log({f"losses/loss-{ii}": loss for ii, loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "As a histogram" %}}
```python
import wandb

# Initialize a new run
with wandb.init(project="log-list-values", name="log-hist") as run:
    # Log losses as a histogram
    losses = [0.1, 0.2, 0.3, 0.4, 0.5]
    run.log({"losses": wandb.Histogram(losses)})
```
{{% /tab %}}
{{< /tabpane >}}

For more, see [the documentation on logging]({{< relref path="/guides/models/track/log/" lang="ko" >}}).