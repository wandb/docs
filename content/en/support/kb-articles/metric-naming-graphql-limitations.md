---
url: /support/:filename
title: "Why can't I sort or filter metrics with special characters?"
toc_hide: true
type: docs
support:
   - experiments
---
Due to GraphQL limitations, metrics containing special characters (such as commas, hyphens, spaces, or symbols) may not be sortable or filterable in the W&B UI.

## Valid metric names

Metric names must follow these rules:
- **Allowed characters**: Letters (A-Z, a-z), digits (0-9), and underscores (_)
- **Starting character**: Must start with a letter or underscore
- **Pattern**: Should match `/^[_a-zA-Z][_a-zA-Z0-9]*$/`

## Examples

**Valid metric names:**
```python
wandb.log({"accuracy": 0.9, "val_loss": 0.1})
wandb.log({"modelAccuracy": 0.95, "learning_rate": 0.001})
```

**Invalid metric names that may cause sorting issues:**
```python
wandb.log({"acc,val": 0.9})  # Contains comma
wandb.log({"loss-train": 0.1})  # Contains hyphen
wandb.log({"test acc": 0.95})  # Contains space
```

## Recommended solution

Replace special characters with underscores when naming metrics:
- Instead of `"test acc"`, use `"test_acc"`
- Instead of `"loss-train"`, use `"loss_train"`
- Instead of `"acc,val"`, use `"acc_val"`

For more information, see [Metric naming constraints]({{< relref "/guides/models/track/log/#metric-naming-constraints" >}}).