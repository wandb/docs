---
title: "Why am I seeing fewer data points than I logged?"
tags:
   - experiments
---

If you are visualizing your metrics against something other than `Step` on your X-Axis, you might see fewer data points than you expect. This is because we require the metrics to be plotted against one another to be logged at the same `Step` - that is how we keep your metrics synchronized, i.e., we only sample metrics that are logged at the same `Step` while interpolating in between samples.\
\
**Guidelines**\
****\
****We recommend you bundle your metrics into the same `log()` call. If your code looks like this:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

It would be better to log it as:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

Alternatively, you can manually control the step parameter and synchronize your metrics in your own code:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

If the value of `step` is the same in both the calls to `log()`, your metrics will be logged under the same step and be sampled together. Please note that step must be monotonically increasing in each call, otherwise the `step` value is ignored during your call to `log()`.