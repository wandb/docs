---
title: "Why am I seeing fewer data points than I logged?"
toc_hide: true
type: docs
tags:
   - experiments
   - metrics
---
When visualizing metrics against an X-axis other than `Step`, expect to see fewer data points. Metrics must log at the same `Step` to remain synchronized. Only metrics logged at the same `Step` are sampled while interpolating between samples.

**Guidelines**

Bundle metrics into a single `log()` call. For example, instead of:

```python
wandb.log({"Precision": precision})
...
wandb.log({"Recall": recall})
```

Use:

```python
wandb.log({"Precision": precision, "Recall": recall})
```

For manual control over the step parameter, synchronize metrics in the code as follows:

```python
wandb.log({"Precision": precision}, step=step)
...
wandb.log({"Recall": recall}, step=step)
```

Ensure the `step` value remains the same in both `log()` calls for the metrics to log under the same step and sample together. The `step` value must increase monotonically in each call; otherwise, the `step` value is ignored.