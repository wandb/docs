---
url: /support/:filename
title: "Why can't I sort or filter metrics with certain characters?"
toc_hide: true
type: docs
support:
   - experiments
---
Metric names in W&B must follow GraphQL naming conventions to ensure they can be properly sorted and filtered in the UI.

## Valid metric names

{{< readfile file="/_includes/metric-naming-rules.md" >}}

Metrics that don't follow these rules may not be sortable or filterable in the W&B UI.

## Examples

{{< readfile file="/_includes/metric-naming-examples.md" >}}

## Recommended solution

Replace invalid characters with valid characters such as underscores:
- Instead of `"test acc"`, use `"test_acc"`
- Instead of `"loss-train"`, use `"loss_train"`
- Instead of `"acc,val"`, use `"acc_val"`

For more information, see [Metric naming constraints]({{< relref "/guides/models/track/log/#metric-naming-constraints" >}}).