---
data_type_classification: class
menu:
  reference:
    identifier: ko-ref-python-automations-metricchangefilter
object_type: automations_namespace
title: MetricChangeFilter
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/automations/_filters/run_metrics.py >}}



Defines a filter that compares a change in a run metric against a user-defined threshold.

The change is calculated over "tumbling" windows, i.e. the difference
between the current window and the non-overlapping prior window.

Attributes:
- agg (Optional): Aggregate operation, if any, to apply over the window size.
- change_dir (ChangeDir): No description provided.
- change_type (ChangeType): No description provided.
- name (str): Name of the observed metric.
- prior_window (int): Size of the prior window over which the metric is aggregated (ignored if `agg is None`).
    If omitted, defaults to the size of the current window.
- threshold (Union): Threshold value to compare against.
- window (int): Size of the window over which the metric is aggregated (ignored if `agg is None`).