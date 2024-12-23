---
title: "How do I log a list of values?"
toc_hide: true
type: docs
tags:
  - logs
  - experiments
---
These examples show logging losses a couple of different ways using [`wandb.log()`](/ref/python/log/).

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

For more, see [the documentation on logging](/guides/track/log/).