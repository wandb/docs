---
description: Visualize and analyze W&B Tables.
menu:
  default:
    identifier: visualize-tables
    parent: tables
title: Visualize and analyze tables
weight: 2
---

Customize your W&B Tables to answer questions about your machine learning model's performance, analyze your data, and more. 

Interactively explore your data to:

* Compare changes precisely across models, epochs, or individual examples
* Understand higher-level patterns in your data
* Capture and communicate your insights with visual samples



{{% alert %}}
W&B Tables posses the following behaviors:
1. **Stateless in an artifact context**: any table logged alongside an artifact version resets to its default state after you close the browser window
2. **Stateful in a workspace or report context**: any changes you make to a table in a single run workspace, multi-run project workspace, or Report persists.

For information on how to save your current W&B Table view, see [Save your view](#save-your-view).
{{% /alert %}}

## How to view two tables
Compare two tables with a [merged view](#merged-view) or a [side-by-side view](#side-by-side-view). For example, the image below demonstrates a table comparison of MNIST data.

{{< img src="/images/data_vis/table_comparison.png" alt="Left: mistakes after 1 training epochs, Right: mistakes after 5 epochs" max-width="90%" >}}

Follow these steps to compare two tables:

1. Go to your project in the W&B App.
2. Select the artifacts icon on the left panel.
2. Select an artifact version. 

In the following image we demonstrate a model's predictions on MNIST validation data after each of five epochs ([view interactive example here](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

{{< img src="/images/data_vis/preds_mnist.png" alt="Click on 'predictions' to view the Table" max-width="90%" >}}


3. Hover over the second artifact version you want to compare in the sidebar and click **Compare** when it appears. For example, in the image below we select a version labeled as "v4" to compare to MNIST predictions made by the same model after 5 epochs of training. 

{{< img src="/images/data_vis/preds_2.png" alt="Preparing to compare model predictions after training for 1 epoch (v0, shown here) vs 5 epochs (v4)" max-width="90%" >}}

### Merged view
<!-- To do, add steps -->
Initially you see both tables merged together. The first table selected has index 0 and a blue highlight, and the second table has index 1 and a yellow highlight. [View a live example of merged tables here](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

{{< img src="/images/data_vis/merged_view.png" alt="In the merged view, numerical columns appears as histograms by default" max-width="90%">}}

From the merged view, you can

* **choose the join key**: use the dropdown at the top left to set the column to use as the join key for the two tables. Typically this is the unique identifier of each row, such as the filename of a specific example in your dataset or an incrementing index on your generated samples. Note that it's currently possible to select _any_ column, which may yield illegible tables and slow queries.
* **concatenate instead of join**: select "concatenating all tables" in this dropdown to _union all the rows_ from both tables into one larger Table instead of joining across their columns
* **reference each Table explicitly**: use 0, 1, and \* in the filter expression to explicitly specify a column in one or both table instances
* **visualize detailed numerical differences as histograms**: compare the values in any cell at a glance

### Side-by-side view

<!-- To do -->

To view the two tables side-by-side, change the first dropdown from "Merge Tables: Table" to "List of: Table" and then update the "Page size" respectively. Here the first Table selected is on the left and the second one is on the right. Also, you can compare these tables vertically as well by clicking on the "Vertical" checkbox.

{{< img src="/images/data_vis/side_by_side.png" alt="In the side-by-side view, Table rows are independent of each other." max-width="90%" >}}

* **compare the tables at a glance**: apply any operations (sort, filter, group) to both tables in tandem and spot any changes or differences quickly. For example, view the incorrect predictions grouped by guess, the hardest negatives overall, the confidence score distribution by true label, etc.
* **explore two tables independently**: scroll through and focus on the side/rows of interest


## Compare artifacts
You can also [compare tables across time](#compare-tables-across-time) or [model variants](#compare-tables-across-model-variants). 


### Compare tables across time
Log a table in an artifact for each meaningful step of training to analyze model performance over training time. For example, you could log a table at the end of every validation step, after every 50 epochs of training, or any frequency that makes sense for your pipeline. Use the side-by-side view to visualize changes in model predictions.

{{< img src="/images/data_vis/compare_across_time.png" alt="For each label, the model makes fewer mistakes after 5 training epochs (R) than after 1 (L)" max-width="90%" >}}

For a more detailed walkthrough of visualizing predictions across training time, [see this report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) and this interactive [notebook example](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..).

### Compare tables across model variants

Compare two artifact versions logged at the same step for two different models to analyze model performance across different configurations (hyperparameters, base architectures, and so forth).

For example, compare predictions between a `baseline` and a new model variant, `2x_layers_2x_lr`, where the first convolutional layer doubles from 32 to 64, the second from 128 to 256, and the learning rate from 0.001 to 0.002. From [this live example](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr), use the side-by-side view and filter down to the incorrect predictions after 1 (left tab) versus 5 training epochs (right tab).

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="After 1 epoch, performance is mixed: precision improves for some classes and worsens for others." >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="After 5 epochs, the 'double' variant is catching up to the baseline." >}}
{{% /tab %}}
{{< /tabpane >}}

## Save your view

Tables you interact with in the run workspace, project workspace, or a report automatically saves their view state. If you apply any table operations then close your browser, the table retains the last viewed configuration when you next navigate to the table. 

{{% alert %}}
Tables you interact with in the artifact context remains stateless.
{{% /alert %}}

To save a table from a workspace in a particular state, export it to a W&B Report. To export a table to report:
1. Select the kebob icon (three vertical dots) in the top right corner of your workspace visualization panel.
2. Select either **Share panel** or **Add to report**.

{{< img src="/images/data_vis/share_your_view.png" alt="Share panel creates a new report, Add to report lets you append to an existing report." max-width="90%">}}


## Examples

These reports highlight the different use cases of W&B Tables:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)