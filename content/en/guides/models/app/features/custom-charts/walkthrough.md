---
description: Tutorial of using the custom charts feature in the W&B UI
menu:
  default:
    identifier: walkthrough
    parent: custom-charts
title: 'Tutorial: Use custom charts'
---

Use custom charts to control the data you're loading in to a panel and its visualization.


## 1. Log data to W&B

First, log data in your script. Use [wandb.Run.config]({{< relref "/guides/models/track/config.md" >}}) for single points set at the beginning of training, like hyperparameters. Use [wandb.Run.log()]({{< relref "/guides/models/track/log/" >}}) for multiple points over time, and log custom 2D arrays with `wandb.Table()`. We recommend logging up to 10,000 data points per logged key.

```python
with wandb.init() as run: 

  # Logging a custom table of data
  my_custom_data = [[x1, y1, z1], [x2, y2, z2]]
  run.log(
    {"custom_data_table": wandb.Table(data=my_custom_data, columns=["x", "y", "z"])}
  )
```

[Try a quick example notebook](https://bit.ly/custom-charts-colab) to log the data tables, and in the next step we'll set up custom charts. See what the resulting charts look like in the [live report](https://app.wandb.ai/demo-team/custom-charts/reports/Custom-Charts--VmlldzoyMTk5MDc).

## 2. Create a query

Once you've logged data to visualize, go to your project page and click the **`+`** button to add a new panel, then select **Custom Chart**. You can follow along in the [custom charts demo workspace](https://app.wandb.ai/demo-team/custom-charts).

{{< img src="/images/app_ui/create_a_query.png" alt="Blank custom chart" >}}

### Add a query

1. Click `summary` and select `historyTable` to set up a new query pulling data from the run history.
2. Type in the key where you logged the `wandb.Table()`. In the code snippet above, it was `my_custom_table` . In the [example notebook](https://bit.ly/custom-charts-colab), the keys are `pr_curve` and `roc_curve`.

### Set Vega fields

Now that the query is loading in these columns, they're available as options to select in the Vega fields dropdown menus:

{{< img src="/images/app_ui/set_vega_fields.png" alt="Pulling in columns from the query results to set Vega fields" >}}

* **x-axis:** runSets_historyTable_r (recall)
* **y-axis:** runSets_historyTable_p (precision)
* **color:** runSets_historyTable_c (class label)

## 3. Customize the chart

Now that looks pretty good, but I'd like to switch from a scatter plot to a line plot. Click **Edit** to change the Vega spec for this built in chart. Follow along in the [custom charts demo workspace](https://app.wandb.ai/demo-team/custom-charts).

{{< img src="/images/general/custom-charts-1.png" alt="Custom chart selection" >}}

I updated the Vega spec to customize the visualization:

* add titles for the plot, legend, x-axis, and y-axis (set “title” for each field)
* change the value of “mark” from “point” to “line”
* remove the unused “size” field

{{< img src="/images/app_ui/customize_vega_spec_for_pr_curve.png" alt="PR curve Vega spec" >}}

To save this as a preset that you can use elsewhere in this project, click **Save as** at the top of the page. Here's what the result looks like, along with an ROC curve:

{{< img src="/images/general/custom-charts-2.png" alt="PR curve chart" >}}

## Bonus: Composite Histograms

Histograms can visualize numerical distributions to help us understand larger datasets. Composite histograms show multiple distributions across the same bins, letting us compare two or more metrics across different models or across different classes within our model. For a semantic segmentation model detecting objects in driving scenes, we might compare the effectiveness of optimizing for accuracy versus intersection over union (IOU), or we might want to know how well different models detect cars (large, common regions in the data) versus traffic signs (much smaller, less common regions). In the[ demo Colab](https://bit.ly/custom-charts-colab), you can compare the confidence scores for two of the ten classes of living things.

{{< img src="/images/app_ui/composite_histograms.png" alt="Composite histogram" >}}

To create your own version of the custom composite histogram panel:

1. Create a new Custom Chart panel in your Workspace or Report (by adding a “Custom Chart” visualization). Hit the “Edit” button in the top right to modify the Vega spec starting from any built-in panel type.
2. Replace that built-in Vega spec with my [MVP code for a composite histogram in Vega](https://gist.github.com/staceysv/9bed36a2c0c2a427365991403611ce21). You can modify the main title, axis titles, input domain, and any other details directly in this Vega spec [using Vega syntax](https://vega.github.io/) (you could change the colors or even add a third histogram :)
3. Modify the query in the right hand side to load the correct data from your wandb logs. Add the field `summaryTable` and set the corresponding `tableKey` to `class_scores` to fetch the `wandb.Table` logged by your run. This will let you populate the two histogram bin sets (`red_bins` and `blue_bins`) via the dropdown menus with the columns of the `wandb.Table` logged as `class_scores`. For my example, I chose the `animal` class prediction scores for the red bins and `plant` for the blue bins.
4. You can keep making changes to the Vega spec and query until you’re happy with the plot you see in the preview rendering. Once you’re done, click **Save as** in the top and give your custom plot a name so you can reuse it. Then click **Apply from panel library** to finish your plot.

Here’s what my results look like from a very brief experiment: training on only 1000 examples for one epoch yields a model that’s very confident that most images are not plants and very uncertain about which images might be animals.

{{< img src="/images/general/custom-charts-3.png" alt="Chart configuration" >}}

{{< img src="/images/general/custom-charts-4.png" alt="Chart result" >}}