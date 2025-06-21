---
description: Compare metrics across multiple runs
menu:
  default:
    identifier: run-comparer
    parent: panels
title: Compare run metrics
weight: 70
---

Use the Run Comparer to see differences and similarities across runs in your project. 

## Add a Run Comparer panel

1. Select the **Add panels** button in the top right corner of the page.
1. From the **Evaluation** section, select **Run comparer**.

## Use Run Comparer
Run Comparer shows the configuration and logged metrics for the 10 first visible runs in the project, one column per run.

- To change the runs to compare, you can search, filter, group, or sort the list of runs on the left-hand side. The Run Comparer updates automatically.
- To filter or search for a configuration key, use the search field at the top of the Run Comparer.
- To quickly see differences and hide identical values, toggle **Diff only** at the top of the panel.
- To adjust the column width or row height, use the formatting buttons at the top of the panel.
- To copy any configuration or metric's value, hover your mouse over the value, then click the copy button. The entire value is copied, even if it is too long to display on the screen.

{{% alert %}}
By default, Run Comparer does not differentiate runs with different values for [`job_type`]({{< relref "/ref/python/sdk/actions/functions/init.md" >}}). This means that it is possible to compare runs that are not comparable within a project. For example, you could compare a training run to a model evaluation run. A training run could contain run logs, hyperparameters, training loss metrics, and the model itself. An evaluation run could use the model to check the model's performance on new training data.

When you search, filter, group, or sort the list of runs in the Runs Table, the Run Comparer automatically updates to compare the first 10 runs. Filter or search within the Runs Table to compare similar runs, such as by filtering or sorting the list by `job_type`. Learn more about [filtering runs]({{< relref "/guides/models/track/runs/filter-runs.md" >}}).
{{% /alert %}}
