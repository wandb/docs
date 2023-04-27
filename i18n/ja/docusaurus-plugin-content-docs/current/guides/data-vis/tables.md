---
description: Visualize and analyze W&B Tables.
displayed_sidebar: ja
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Visualize & Analyze Tables

Use W&B Tables to log and visualize data and model predictions. Interactively explore your data to:

* Compare changes precisely across models, epochs, or individual examples
* Understand higher-level patterns in your data
* Capture and communicate your insights with visual samples

Customize your W&B Tables to answer questions about your machine learning model's performance, analyze your data, and more. 

:::info
W&B Tables posses the following behaviors:
1. **stateless in an artifact context**: any Table logged alongside an artifact version will reset to its default state after you close the browser window
2. **stateful in a workspace or report context**: any changes you make to a Table in a single run workspace, multi-run project workspace, or Report will persist.

For information on how to save your current W&B Table view, see [Save your view](#save-your-view).
:::


## Table operations

Use the W&B App to sort, filter, and group your W&B Tables. 

<!-- [Try these yourself →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json) -->

### Sort

Sort all rows in a Table by the value in a given column. 
1. Hover your mouse over the column title. A kebob menu will appear (three vertical docs).
2. Select on the kebob menu (three vertical dots).
3. Choose **Sort Asc** or **Sort Desc** to sort the rows in ascending or descending order, respectively. 

![See the digits for which the model most confidently guessed "0".](/images/data_vis/data_vis_sort_kebob.png)

The preceding image demonstrates how to view sorting options for a Table column called `val_acc`.

### Filter

Filter all rows by an expression with the **Filter** button on the top left of the dashboard. 

![See only examples which the model gets wrong.](/images/data_vis/filter.png)

Select **Add filter** to add one or more filters to your rows. Three dropdown menus will appear. From left to right the filter types are based on: Column name, Operator , and Values

|                   | Column name | Binary relation    | Value       |
| -----------       | ----------- | ----------- | ----------- |
| Accepted values   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |


The expression editor shows a list of options for each term using autocomplete on column names and logical predicate structure. You can connect multiple logical predicates into one expression using "and" or "or" (and sometimes parentheses).

![](/images/data_vis/filter_example.png)
The preceding image demonstrates shows a filter that is based on the `val_loss` column. The filter shows W&B Runs with a validation loss less than or equal to 1.


### Group

Group all rows by the value in a particular column with the **Group by** button in a column header. 

![The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s.](/images/data_vis/group.png)

By default, this turns other numeric columns into histograms showing the distribution of values for that column across the group. Grouping is helpful for understanding higher-level patterns in your data.

<!-- ## Modify columns
The proceeding sections demonstrate how to modify W&B Tables. -->

<!-- ### Add columns

You can insert a new column to the left or right of a Table. To add a column, select the kebob menu

From the kebob menu on any column, you can insert a new column to the left or right. Edit the cell expression to compute a new column using references to existing columns, mathematical and logical operators, and aggregation functions when a row is grouped (like average, sum, min/max). Optionally give the column a new name below the expression editor.

![The closed\_loop\_score column sums the confidence scores for digits with typical loops (0, 6, 8, 9).](/images/data_vis/add_columns.png) -->

<!-- ### Edit columns and display settings

Tables render column data based on the type of the values logged in that column. By clicking on the column name or "Column settings" from the three-dot menu, you can modify

* **the contents** of the column by editing "Cell expression": select a different field to show, or build a logical predicate expression as described above, including adding a function like count() or avg(), etc to apply to the contents.
* **the column type**: convert between a histogram, an array of values, a number, text, etc. W&B will try to guess the type based on the data contents.
* **the pagination**: select how many objects to view at once in a grouped row
* **the display name** in the column header

### Remove columns -->

<!-- Select "Remove" to delete a column. -->

## Compare tables

All the operations described above also work in the context of Table comparison.

![Left: mistakes after 1 training epochs, Right: mistakes after 5 epochs](/images/data_vis/table_comparison.png)

### From the UI

To compare two Tables, start by viewing one Table logged alongside an artifact. In the following image we demonstrate a model's predictions on MNIST validation data after each of five epochs ([interactive example →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json))

![Click on "predictions" to view the Table](@site/static/images/data_vis/preds_mnist.png)

Next, select a different artifact version for comparison—for example, "v4" to compare to MNIST predictions made by the same model after 5 epochs of training. Hover over the second artifact version in the sidebar and click "Compare" when it appears.

![Preparing to compare model predictions after training for 1 epoch (v0, shown here) vs 5 epochs (v4)](@site/static/images/data_vis/preds_2.png)

#### Merged view

[Live example →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#7dd0cd845c0edb469dec)

Initially you will see both Tables merged together. The first Table selected has index 0 and a blue highlight, and the second Table has index 1 and a yellow highlight.

![In the merged view, numerical columns will appear as histograms by default](@site/static/images/data_vis/merged_view.png)

From the merged view, you can

* **choose the join key**: use the dropdown at the top left to set the column to use as the join key for the two tables. Typically this will be the unique identifier of each row, such as the file name of a specific example in your dataset or an incrementing index on your generated samples. Note that it's currently possible to select _any_ column, which may yield illegible Tables and slow queries.
* **concatenate instead of join**: select "concatenating all tables" in this dropdown to _union all the rows_ from both Tables into one larger Table instead of joining across their columns
* **reference each Table explicitly**: use 0, 1, and \* in the filter expression to explicitly specify a column in one or both Table instances
* **visualize detailed numerical differences as histograms**: compare the values in any cell at a glance

#### Side-by-side view

To view the two Tables side-by-side, change the first dropdown from "Merge Tables: Table" to "List of: Table" and then update the "Page size" respectively. Here the first Table selected is on the left and the second one is on the right. Also, you can compare these tables vertically as well by clicking on the "Vertical" checkbox.

![In the side-by-side view, Table rows are independent of each other.](/images/data_vis/side_by_side.png)

* **compare the Tables at a glance**: apply any operations (sort, filter, group) to both Tables in tandem and spot any changes or differences quickly. For example, view the incorrect predictions grouped by guess, the hardest negatives overall, the confidence score distribution by true label, etc.
* **explore two Tables independently**: scroll through and focus on the side/rows of interest

### Compare across time

To analyze model performance over training time, log a Table in an artifact context for each meaningful step of training: at the end of every validation step, after every 50 epochs of training, or any frequency that makes sense for your pipeline. Use the side-by-side view to visualize changes in model predictions.

![For each label, the model makes fewer mistakes after 5 training epochs (R) than after 1 (L)](/images/data_vis/compare_across_time.png)

For a more detailed walkthrough of visualizing predictions across training time, [see this report](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk) and this interactive [notebook example →](http://wandb.me/tables-quickstart)

### Compare across model variants

To analyze model performance across different configurations (hyperparameters, base architectures, etc), compare two artifact versions logged at the same step for two different models. For example, compare predictions between a `baseline` and a new model variant, `2x_layers_2x_lr`, where the first convolutional layer doubles from 32 to 64, the second from 128 to 256, and the learning rate from 0.001 to 0.002. From [this live example](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json#2bb3b1d40aa777496b5d$2x\_layers\_2x\_lr), use the side-by-side view and filter down to the incorrect predictions after 1 (left tab) versus 5 training epochs (right tab).

This is a toy example of model comparison, but it illustrates the ease, flexibility, and depth of the exploratory analysis you can do with Tables—without rerunning any of your code, writing new one-off scripts, generating new charts, etc.

<Tabs
  defaultValue="one_epoch"
  values={[
    {label: '1 training epoch', value: 'one_epoch'},
    {label: '5 training epochs', value: 'five_epochs'},
  ]}>
  <TabItem value="one_epoch">

![After 1 epoch, performance is mixed: precision improves for some classes and worsens for others.](/images/data_vis/compare_across_variants.png)
  </TabItem>
  <TabItem value="five_epochs">

![After 5 epochs, the "double" variant is catching up to the baseline.](/images/data_vis/compare_across_variants_after_5_epochs.png)
  </TabItem>
</Tabs>


## Save your view

Tables you interact with in the run workspace, project workspace, or a report will automatically save their view state. If you apply any Table operations then close your browser, the Table will retain the last viewed configuration when you next navigate to the table.

Tables you interact with in the artifact context will remain stateless.

To save a Table from a workspace in a particular state, export it to a Report. You can do this from the three dot menu in the top right corner of any workspace visualization panel (three dots → "Share panel" or "Add to report").

![Share panel creates a new report, Add to report lets you append to an existing report.](/images/data_vis/share_your_view.png)


### Examples

These reports highlight the different use cases of W&B Tables:

* [Visualize Predictions Over Time](https://wandb.ai/stacey/mnist-viz/reports/Visualize-Predictions-over-Time--Vmlldzo1OTQxMTk)
* [How to Compare Tables in Workspaces](https://wandb.ai/stacey/xtable/reports/How-to-Compare-Tables-in-Workspaces--Vmlldzo4MTc0MTA)
* [Image & Classification Models](https://wandb.ai/stacey/mendeleev/reports/Tables-Tutorial-Visualize-Data-for-Image-Classification--VmlldzozNjE3NjA)
* [Text & Generative Language Models](https://wandb.ai/stacey/nlg/reports/Tables-Tutorial-Visualize-Text-Data-Predictions---Vmlldzo1NzcwNzY)
* [Named Entity Recognition](https://wandb.ai/stacey/ner\_spacy/reports/Named-Entity-Recognition--Vmlldzo3MDE3NzQ)
* [AlphaFold Proteins](https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc)
