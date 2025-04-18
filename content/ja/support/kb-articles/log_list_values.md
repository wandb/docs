---
menu:
  support:
    identifier: ja-support-kb-articles-log_list_values
support:
- logs
- experiments
title: How do I log a list of values?
toc_hide: true
type: docs
url: /support/:filename
---

These examples show logging losses a couple of different ways using [`wandb.log()`]({{< relref path="/ref/python/log/" lang="ja" >}}).

{{< tabpane text=true >}}
{{% tab "Using a dictionary" %}}
```python
wandb.log({f"losses/loss-{ii}": loss for ii, 
  loss in enumerate(losses)})
```
{{% /tab %}}
{{% tab "As a histogram" %}}
```python
# Converts losses to a histogram
wandb.log({"losses": wandb.Histogram(losses)})  
```
{{% /tab %}}
{{< /tabpane >}}

For more, see [the documentation on logging]({{< relref path="/guides/models/track/log/" lang="ja" >}}).