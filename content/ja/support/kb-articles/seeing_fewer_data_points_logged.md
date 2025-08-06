---
menu:
  support:
    identifier: ja-support-kb-articles-seeing_fewer_data_points_logged
support:
- experiments
- metrics
title: Why am I seeing fewer data points than I logged?
toc_hide: true
type: docs
url: /support/:filename
---

When visualizing metrics against an X-axis other than `Step`, expect to see fewer data points. Metrics must log at the same `Step` to remain synchronized. Only metrics logged at the same `Step` are sampled while interpolating between samples.

**Guidelines**

Bundle metrics into a single `log()` call. For example, instead of:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision})
    ...
    run.log({"Recall": recall})
```

Use:

```python
import wandb
with wandb.init() as run:
    run.log({"Precision": precision, "Recall": recall})
```

For manual control over the step parameter, synchronize metrics in the code as follows:

```python
with wandb.init() as run:
    step = 100  # Example step value
    # Log Precision and Recall at the same step
    run.log({"Precision": precision, "Recall": recall}, step=step)
```

Ensure the `step` value remains the same in both `log()` calls for the metrics to log under the same step and sample together. The `step` value must increase monotonically in each call. Otherwise, the `step` value is ignored.