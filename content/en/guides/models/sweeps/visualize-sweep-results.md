---
description: Visualize the results of your W&B Sweeps with the W&B App UI.
menu:
  default:
    identifier: visualize-sweep-results
    parent: sweeps
title: Visualize sweep results
weight: 7
---

Visualize the results of your W&B Sweeps with the W&B App. Navigate to the [W&B App](https://wandb.ai/home). Choose the project that you specified when you initialized a sweep. You will be redirected to your project [workspace]({{< relref "/guides/models/track/workspaces.md" >}}). Select the **Sweep icon** on the left panel (broom icon). From the Sweep UI, select the name of your Sweep from the list.

By default, W&B will automatically create a parallel coordinates plot, a parameter importance plot, and a scatter plot when you start a W&B Sweep job.

{{< img src="/images/sweeps/navigation_sweeps_ui.gif" alt="Sweep UI navigation" >}}

Parallel coordinates charts summarize the relationship between large numbers of hyperparameters and model metrics at a glance. For more information on parallel coordinates plots, see [Parallel coordinates]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}}).

{{< img src="/images/sweeps/example_parallel_coordiantes_plot.png" alt="Example parallel coordinates plot." >}}

The scatter plot(left) compares the W&B Runs that were generated during the Sweep. For more information about scatter plots, see [Scatter Plots]({{< relref "/guides/models/app/features/panels/scatter-plot.md" >}}).

The parameter importance plot(right) lists the hyperparameters that were the best predictors of, and highly correlated to desirable values of your metrics. For more information on parameter importance plots, see [Parameter Importance]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}}).

{{< img src="/images/sweeps/scatter_and_parameter_importance.png" alt="Scatter plot and parameter importance" >}}


You can alter the dependent and independent values (x and y axis) that are automatically used. Within each panel there is a pencil icon called **Edit panel**. Choose **Edit panel**. A model will appear. Within the modal, you can alter the behavior of the graph.

For more information on all default W&B visualization options, see [Panels]({{< relref "/guides/models/app/features/panels/" >}}). See the [Data Visualization docs]({{< relref "/guides/models/tables/" >}}) for information on how to create plots from W&B Runs that are not part of a W&B Sweep.