---
description: Try visualizing data and model predictions in 5 minutes
---

# Tables Quickstart

Explore how to use W&B Tables with this 5 minute quickstart, which runs through how to log data tables, then visualize and query that data. Click the button below to try a PyTorch quickstart example project on MNIST data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/tables-quickstart)

## 1. Log a table

Initialize a run, create a `wandb.Table()`, then log it to the run.

```python
run = wandb.init(project="table-test")
my_table = wandb.Table(columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]])
run.log({"Table Name": my_table})
```

## 2. Visualize tables in the workspace

See the resulting table in the workspace. A new panel is added for each unique table key. In the above example, `my_table` is logged under the key `Table Name`, which creates the displayed table below:

![](/images/data_vis/wandb_demo_logged_sample_table.png)

## 3. Compare across model versions

Log sample tables from multiple different runs, then compare results in the project workspace. In this [example workspace](https://wandb.ai/carey/table-test?workspace=user-carey), we show how to combine rows from multiple different versions in the same table.

![](/images/data_vis/wandb_demo_toggle_on_and_off_cross_run_comparisons_in_tables.gif)

Use the table filter, sort, and grouping features to explore and evaluate model results.

![](/images/data_vis/wandb_demo_filter_on_a_table.png)
