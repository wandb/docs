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

For information on how to save your current W&B Table view, see [Save your view]({{< relref "#save-your-view" >}}).
{{% /alert %}}

## Compare two tables
Compare two tables with a [merged view]({{< relref "#merged-view" >}}) or a [side-by-side view]({{< relref "#side-by-side-view" >}}). For example, the image below demonstrates a table comparison of MNIST data.

{{< img src="/images/data_vis/table_comparison.png" alt="Training epoch comparison" max-width="90%" >}}

Follow these steps to compare two tables:

1. Go to your project in the W&B App.
2. Select the artifacts icon on the left panel.
2. Select an artifact version. 

In the following image we demonstrate a model's predictions on MNIST validation data after each of five epochs ([view interactive example here](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json)).

{{< img src="/images/data_vis/preds_mnist.png" alt="Click on 'predictions' to view the Table" max-width="90%" >}}


3. Hover over the second artifact version you want to compare in the sidebar and click **Compare** when it appears. For example, in the image below we select a version labeled as "v4" to compare to MNIST predictions made by the same model after 5 epochs of training. 

{{< img src="/images/data_vis/preds_2.png" alt="Model prediction comparison" max-width="90%" >}}

### Merged view
<!-- To do, add steps -->
Initially you see both tables merged together. The first table selected has index 0 and a blue highlight, and the second table has index 1 and a yellow highlight. [View a live example of merged tables here](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec).

{{< img src="/images/data_vis/merged_view.png" alt="Merged view" max-width="90%">}}

From the merged view, you can

* **choose the join key**: use the dropdown at the top left to set the column to use as the join key for the two tables. Typically this is the unique identifier of each row, such as the filename of a specific example in your dataset or an incrementing index on your generated samples. Note that it's currently possible to select _any_ column, which may yield illegible tables and slow queries.
* **concatenate instead of join**: select "concatenating all tables" in this dropdown to _union all the rows_ from both tables into one larger Table instead of joining across their columns
* **reference each Table explicitly**: use 0, 1, and \* in the filter expression to explicitly specify a column in one or both table instances
* **visualize detailed numerical differences as histograms**: compare the values in any cell at a glance

### Side-by-side view

<!-- To do -->

To view the two tables side-by-side, change the first dropdown from "Merge Tables: Table" to "List of: Table" and then update the "Page size" respectively. Here the first Table selected is on the left and the second one is on the right. Also, you can compare these tables vertically as well by clicking on the "Vertical" checkbox.

{{< img src="/images/data_vis/side_by_side.png" alt="Side-by-side table view" max-width="90%" >}}

* **compare the tables at a glance**: apply any operations (sort, filter, group) to both tables in tandem and spot any changes or differences quickly. For example, view the incorrect predictions grouped by guess, the hardest negatives overall, the confidence score distribution by true label, etc.
* **explore two tables independently**: scroll through and focus on the side/rows of interest



## Visualize how values change throughout your runs

View how values you log to a table change throughout your runs with a step slider. Slide the step slider to view the values logged at different steps. For example, you can view how the loss, accuracy, or other metrics change after each run. 

The slider uses a key to determine the step value. The default key for the slider is `_step`, a special key that W&B automatically logs for you. The `_step` key is an integer that increments by 1 each time you call `wandb.Run.log()` in your code.

To add a step slider to a W&B Table:

1. Navigate to your project's workspace.
2. Click **Add panel** in the top right corner of the workspace.
3. Select **Query panel**.
4. Within the query expression editor, select `runs` and press **Enter** on your keyboard.
5. Click the gear icon to view the settings for the panel.
6. Set **Render As** selector to **Stepper**.
7. Set **Stepper Key** to `_step` or the [key to use as the unit]({{< relref "#custom-step-keys" >}}) for the step slider.

The following image shows a query panel with three W&B runs and the values they logged at step 295.

{{< img src="/images/data_vis/stepper_key.png" alt="Step slider feature">}}

Within the W&B App UI you may notice duplicate values for multiple steps. This duplication can occur if multiple runs log the same value at different steps, or if a run does not log values at every step. If a value is missing for a given step, W&B uses the last value that was logged as the slider key.

### Custom step key

The step key can be any numeric metric that you log in your runs as the step key, such as `epoch` or `global_step`. When you use a custom step key, W&B maps each value of that key to a step (`_step`) in the run.

This table shows how a custom step key `epoch` maps to `_step` values for three different runs: `serene-sponge`, `lively-frog`, and `vague-cloud`. Each row represents a call to `wandb.Run.log()` at a particular `_step` in a run. The columns show the corresponding epoch values, if any, that were logged at those steps. Some `_step` values are omitted to save space.

The first time `wandb.Run.log()` was called, none of the runs logged an `epoch` value, so the table shows empty values for `epoch`. 

| `_step` | vague-cloud (`epoch`) | lively-frog(`epoch`) |  serene-sponge (`epoch`) |
| ------- | ------------- | ----------- | ----------- |
| 1 | | |  |
| 2  |   |   | 1  | 
| 4  |   | 1 | 2  |
| 5  | 1 |   |  |
| 6  |  |   | 3  |
| 8  |  | 2 | 4  |
| 10 |  |   | 5  |
| 12 |  | 3 | 6  |
| 14 |  |   |  7 | 
| 15 | 2  |   |  |
| 16 |  | 4 | 8  | 
| 18 |  |   | 9  |
| 20 | 3 | 5 | 10 |

Now, if the slider is set to `epoch = 1`, the following happens:

* `vague-cloud` finds `epoch = 1` and returns the value logged at `_step = 5`
* `lively-frog` finds `epoch = 1` and returns the value logged at `_step = 4`
* `serene-sponge` finds `epoch = 1` and returns the value logged at `_step = 2`

If the slider is set to `epoch = 9`:

* `vague-cloud` also doesn't log `epoch = 9`, so W&B uses the latest prior value `epoch = 3` and returns the value logged at `_step = 20`
* `lively-frog` doesn’t log `epoch = 9`, but the latest prior value is `epoch = 5` so it returns the value logged at `_step = 20`
* `serene-sponge` finds `epoch = 9` and return the value logged at `_step = 18`

<!-- | | |
| ---- | ---- |
| Run History Tables Stepper | |
| Run History Plots Stepper | |
| Stepper | | -->

## Compare artifacts
You can also [compare tables across time]({{< relref "#compare-tables-across-time" >}}) or [model variants]({{< relref "#compare-tables-across-model-variants" >}}). 


### Compare tables across time
Log a table in an artifact for each meaningful step of training to analyze model performance over training time. For example, you could log a table at the end of every validation step, after every 50 epochs of training, or any frequency that makes sense for your pipeline. Use the side-by-side view to visualize changes in model predictions.

{{< img src="/images/data_vis/compare_across_time.png" alt="Training progress comparison" max-width="90%" >}}

For a more detailed walkthrough of visualizing predictions across training time, see the [predictions over time report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) and this interactive [notebook example](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb?_gl=1*kf20ui*_gcl_au*OTI3ODM1OTcyLjE3MzE0MzU1NjU.*_ga*ODEyMjQ4MjkyLjE3MzE0MzU1NjU.*_ga_JH1SJHJQXJ*MTczMTcwNTMwNS45LjEuMTczMTcwNTM5My4zMy4wLjA.*_ga_GMYDGNGKDT*MTczMTcwNTMwNS44LjEuMTczMTcwNTM5My4wLjAuMA..).

### Compare tables across model variants

Compare two artifact versions logged at the same step for two different models to analyze model performance across different configurations (hyperparameters, base architectures, and so forth).

For example, compare predictions between a `baseline` and a new model variant, `2x_layers_2x_lr`, where the first convolutional layer doubles from 32 to 64, the second from 128 to 256, and the learning rate from 0.001 to 0.002. From [this live example](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x_layers_2x_lr), use the side-by-side view and filter down to the incorrect predictions after 1 (left tab) versus 5 training epochs (right tab).

{{< tabpane text=true >}}
{{% tab header="1 training epoch" value="one_epoch" %}}
{{< img src="/images/data_vis/compare_across_variants.png" alt="Performance comparison" >}}
{{% /tab %}}
{{% tab header="5 training epochs" value="five_epochs" %}}
{{< img src="/images/data_vis/compare_across_variants_after_5_epochs.png" alt="Variant performance comparison" >}}
{{% /tab %}}
{{< /tabpane >}}

## Save your view

Tables you interact with in the run workspace, project workspace, or a report automatically saves their view state. If you apply any table operations then close your browser, the table retains the last viewed configuration when you next navigate to the table. 

{{% alert %}}
Tables you interact with in the artifact context remains stateless.
{{% /alert %}}

To save a table from a workspace in a particular state, export it to a W&B Report. To export a table to report:
1. Select the kebab icon (three vertical dots) in the top right corner of your workspace visualization panel.
2. Select either **Share panel** or **Add to report**.

{{< img src="/images/data_vis/share_your_view.png" alt="Report sharing options" max-width="90%">}}


## Examples

These reports highlight the different use cases of W&B Tables:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)