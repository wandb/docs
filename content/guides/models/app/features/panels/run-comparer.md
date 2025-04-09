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
Run Comparer does not differentiate runs with different values for [`job_type`]({{< relref "/ref/python/init.md" >}}). This means that it is possible to compare runs that are not really comparable, like comparing an image run to an audio run. Search, filter, group, or sort the list of runs to limit it to the runs you want to analyze. For example, filter or sort the list of runs by `job_type` to compare similar runs.
{{% /alert %}}
