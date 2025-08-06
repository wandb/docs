---
description: Explore how to use W&B Tables with this 5 minute Quickstart.
menu:
  default:
    identifier: ja-guides-models-tables-tables-walkthrough
    parent: tables
title: 'Tutorial: Log tables, visualize and query data'
weight: 1
---

The following Quickstart demonstrates how to log data tables, visualize data, and query data.

Select the button below to try a PyTorch Quickstart example project on MNIST data. 

## 1. Log a table
Log a table with W&B. You can either construct a new table or pass a Pandas Dataframe.

{{< tabpane text=true >}}
{{% tab header="Construct a table" value="construct" %}}
To construct and log a new Table, you will use:
- [`wandb.init()`]({{< relref path="/ref/python/sdk/functions/init.md" lang="ja" >}}): Create a [run]({{< relref path="/guides/models/track/runs/" lang="ja" >}}) to track results.
- [`wandb.Table()`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}): Create a new table object.
  - `columns`: Set the column names.
  - `data`: Set the contents of each row.
- [`wandb.Run.log()`]({{< relref path="/ref/python/sdk/classes/run.md/#method-runlog" lang="ja" >}}): Log the table to save it to W&B.

Here's an example:
```python
import wandb

with wandb.init(project="table-test") as run:
    # Create and log a new table.
    my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
    run.log({"Table Name": my_table})
```
{{% /tab %}}

{{% tab header="Pandas Dataframe" value="pandas"%}}
Pass a Pandas Dataframe to `wandb.Table()` to create a new table.

```python
import wandb
import pandas as pd

df = pd.read_csv("my_data.csv")

with wandb.init(project="df-table") as run:
    # Create a new table from the DataFrame
    # and log it to W&B.
  my_table = wandb.Table(dataframe=df)
  run.log({"Table Name": my_table})
```

For more information on supported data types, see the [`wandb.Table`]({{< relref path="/ref/python/sdk/data-types/table.md" lang="ja" >}}) in the W&B API Reference Guide.
{{% /tab %}}
{{< /tabpane >}}


## 2. Visualize tables in your project workspace

View the resulting table in your workspace. 

1. Navigate to your project in the W&B App.
2. Select the name of your run in your project workspace. A new panel is added for each unique table key. 

{{< img src="/images/data_vis/wandb_demo_logged_sample_table.png" alt="Sample table logged" >}}

In this example, `my_table`, is logged under the key `"Table Name"`.

## 3. Compare across model versions

Log sample tables from multiple W&B Runs and compare results in the project workspace. In this [example workspace](https://wandb.ai/carey/table-test?workspace=user-carey), we show how to combine rows from multiple different versions in the same table.

{{< img src="/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif" alt="Cross-run table comparison" >}}

Use the table filter, sort, and grouping features to explore and evaluate model results.

{{< img src="/images/data_vis/wandb_demo_filter_on_a_table.png" alt="Table filtering" >}}