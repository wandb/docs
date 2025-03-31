---
description: Compare results across machine learning experiments
menu:
  default:
    identifier: ja-guides-models-app-features-panels-parallel-coordinates
    parent: panels
title: Parallel coordinates
weight: 30
---

Parallel coordinates charts summarize the relationship between large numbers of hyperparameters and model metrics at a glance.

{{< img src="/images/app_ui/parallel_coordinates.gif" alt="" >}}

* **Axes**: Different hyperparameters from [`wandb.config`]({{< relref path="/guides/models/track/config.md" lang="ja" >}}) and metrics from [`wandb.log`]({{< relref path="/guides/models/track/log/" lang="ja" >}}).
* **Lines**: Each line represents a single run. Mouse over a line to see a tooltip with details about the run. All lines that match the current filters will be shown, but if you turn off the eye, lines will be grayed out.

## Create a parallel coordinates panel

1. Go to the landing page for your workspace
2. Click **Add Panels**
3. Select **Parallel coordinates**

## Panel Settings

To configure the panel, click the edit button in the upper right corner of the panel.

* **Tooltip**: On hover, a legend shows up with info on each run
* **Titles**: Edit the axis titles to be more readable
* **Gradient**: Customize the gradient to be any color range you like
* **Log scale**: Each axis can be set to view on a log scale independently
* **Flip axis**: Switch the axis direction— this is useful when you have both accuracy and loss as columns

[Interact with a live parallel coordinates panel](https://app.wandb.ai/example-team/sweep-demo/reports/Zoom-in-on-Parallel-Coordinates-Charts--Vmlldzo5MTQ4Nw)