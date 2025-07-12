---
title: MetricThresholdFilter
object_type: automations_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



Defines a filter that compares a run metric against a user-defined threshold value.

Attributes:
- agg (Optional): Aggregate operation, if any, to apply over the window size.
- cmp (Literal): Comparison operator used to compare the metric value (left) vs. the threshold value (right).
- name (str): Name of the observed metric.
- threshold (Union): Threshold value to compare against.
- window (int): Size of the window over which the metric is aggregated (ignored if `agg is None`).