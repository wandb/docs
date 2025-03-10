---
title: "Why is the same metric appearing more than once?"
toc_hide: true
type: docs
tags:
   - experiments
---
When logging various data types under the same key, split them in the database. This results in multiple entries of the same metric name in the UI dropdown. The data types grouped are `number`, `string`, `bool`, `other` (primarily arrays), and any `wandb` data type such as `Histogram` or `Image`. Send only one type per key to prevent this issue.

Metric names are case-insensitive. Avoid using names that differ only by case, such as `"My-Metric"` and `"my-metric"`.