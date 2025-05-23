---
title: Log tables
weight: 1
---

Visualize and log tabular data with W&B Tables. A table is a two-dimensional grid of data where each column has a single type of data. Each row represents one or more data points logged to a W&B [run]({{< relref "/guides/models/track/runs/" >}}). Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types.

Tables are a specialized [data type]({{< relref "/ref/python/data-types/" >}}) in W&B that are logged internally in W&B as [artifact]({{< relref "/guides/core/artifacts/" >}}) objects.

You [create and log tables]({{< relref "#create-and-log-a-new-table" >}}) using the W&B Python SDK. When you create a table object, you specify the columns and data for the table and a [mode]({{< relref "#table-logging-modes" >}}). The mode determines how the table is logged and updated during your ML experiments.


## Create and log a table


1. First, create a new run with `wandb.init()`. 
2. Next, create a table object with the [`wandb.Table`]({{< relref "/ref/python/data-types/table" >}}) Class. Specify the columns and data for the table for the `columns` and `data` parameters, respectively. Though optional, it is recommended to set the `log_mode` parameter to one of the three modes: `IMMUTABLE`, `MUTABLE`, or `INCREMENTAL`. The default mode is `IMMUTABLE`. See [Table Logging Modes]({{< relref "#table-logging-modes" >}}) in the next section for more information.
3. Lastly, log the table to W&B with `run.log()`.

The following example shows how to create and log a table with two columns, `a` and `b`, and two rows of data, `["a1", "b1"]` and `["a2", "b2"]`:

```python
import wandb

# Start a new run
run = wandb.init(project="table-demo")

# Create a table object with two columns and two rows of data
my_table = wandb.Table(
    columns=["a", "b"],
    data=[["a1", "b1"], ["a2", "b2"]],
    log_mode="IMMUTABLE"
    )

# Log the table to W&B
run.log({"Table Name": my_table})

# Finish the run
run.finish()
```

## Logging modes

The `log_mode` parameter of the [`wandb.Table`]({{< relref "/ref/python/data-types/table" >}}) class determines how the table is logged and updated during your ML experiments. The `log_mode` parameter accepts three values: `IMMUTABLE`, `MUTABLE`, and `INCREMENTAL`. Each mode has different implications for how the table is logged, how it can be modified, and how it is rendered in the W&B App.

The modes are:
- `IMMUTABLE`: Once a table is logged to W&B, you can not modify it.
- `MUTABLE`: After you log the table to W&B, you can overwrite the existing table by replacing that table with a new one.
- `INCREMENTAL`: Add batches of new rows to a table throughout the machine learning experiment.

The following table describes the differences between the three modes and common use cases for each:

| Mode  | Use Cases  | Benefits  |
| ----- | ---------- | ----------|
| `IMMUTABLE`   | - Storing tabular data generated at the end of a run for further analysis                              | - Minimal overhead when logged at the end of a run<br>- All rows rendered in UI |
| `MUTABLE`     | - Adding columns or rows to existing tables<br>- Enriching results with new information                        | - Capture Table mutations<br>- All rows rendered in UI                          |
| `INCREMENTAL` | - Adding rows to tables in batches<br> - Long-running training jobs<br>- Processing large datasets in batches<br>- Monitoring ongoing results | - View updates on UI during training<br>- Ability to step through increments    |

The next sections show example code snippets for each mode along with considerations for when to use each mode.

### MUTABLE mode

`MUTABLE` mode updates an existing table by replacing the existing table with a new one. `MUTABLE` mode is useful when you want to add new columns and rows to an existing table in a non iterative process. If you want to add new batches of rows incrementally like in a training loop, consider using [`INCREMENTAL` mode]({{< relref "#INCREMENTAL-mode" >}}) instead.

The following example shows how to create a table in `MUTABLE` mode, log it, and then add new columns to it.

{{% alert %}}
The following example uses a placeholder function `load_eval_data()` to load data and a placeholder function `model.predict()` to make predictions. You will need to replace these with your own data loading and prediction functions.
{{% /alert %}}

```python
import wandb
import numpy as np

run = wandb.init(project="mutable-table-demo")

# Create a table object with MUTABLE logging mode
table = wandb.Table(columns=["input", "label", "prediction"],
                    log_mode="MUTABLE")

# Load data and make predictions
inputs, labels = load_eval_data() # Placeholder function
raw_preds = model.predict(inputs) # Placeholder function

for inp, label, pred in zip(inputs, labels, raw_preds):
    table.add_data(inp, label, pred)

# Step 1: Log initial data 
wandb.log({"eval_table": table})  # Log initial table

# Step 2: Add confidence scores (e.g. max softmax)
confidences = np.max(raw_preds, axis=1)
table.add_column("confidence", confidences)
run.log({"eval_table": table})  # Add confidence info

# Step 3: Add post-processed predictions
# (e.g., thresholded or smoothed outputs)
post_preds = (confidences > 0.7).astype(int)
table.add_column("final_prediction", post_preds)
wandb.log({"eval_table": table})  # Final update with another column

run.finish()
```
In the previous example, the table is logged three times: once with the initial data, once with the confidence scores, and once with the final predictions.

{{% alert %}}
Internally, the table is replaced each time you log the table. Overwriting a table with a new one is computationally expensive and can be slow for large tables.
{{% /alert %}}

### INCREMENTAL mode

In incremental mode, you log batches of rows to a table throughout the machine learning experiment. This
is ideal for monitoring long-running jobs or when working with large tables that would be inefficient to log during the run for updates.

You can not add new columns to a table in `INCREMENTAL` mode. If you need to add new columns to a table that you already logged, consider using `MUTABLE` mode instead.

The following example shows how to create a table in `INCREMENTAL` mode, log it, and then add new rows to it.

{{% alert %}}

The following example uses a placeholder function `get_training_batch()` to load data, a placeholder function `train_model_on_batch()` to train the model, and a placeholder function `predict_on_batch()` to make predictions. You will need to replace these with your own data loading, training, and prediction functions.

{{% /alert %}}

```python
import wandb

run = wandb.init(project="incremental-table-demo")

# Create a table with INCREMENTAL logging mode
table = wandb.Table(columns=["step", "input", "label", "prediction"],
                    log_mode="INCREMENTAL")

# Training loop
for step in range(get_num_batches()): # Placeholder function
    # Load batch data
    inputs, labels = get_training_batch(step) # Placeholder function

    # Train and predict
    train_model_on_batch(inputs, labels) # Placeholder function
    predictions = predict_on_batch(inputs) # Placeholder function

    # Add batch data to table
    for input_item, label, prediction in zip(inputs, labels, predictions):
        table.add_data(step, input_item, label, prediction)

    # Log the table incrementally
    wandb.log({"training_table": table}, step=step)

run.finish()
```

In the previous code example, the table is logged once per training step (`step`).

Incremental logging is generally more computationally efficient than logging a new table each time (`log_mode=MUTABLE`). However, it is important to note that the W&B App may not render all rows in the table if you log a large number of increments. If your goal is to update and view your table data while your run is ongoing and to have all the data available for analysis, consider using both `INCREMENTAL` and `IMMUTABLE` logging modes.

{{% alert %}}
Run workspaces in the W&B App have a limit of 100 increments. If you log more than 100 increments, only the most recent 100 are shown in the run workspace.
{{% /alert %}}

As an example, consider the follow code snippet that combines `INCREMENTAL` and `IMMUTABLE` logging modes:

```python
import wandb

run = wandb.init(project="combined-logging-example")

# Create an incremental table for efficient updates during training
incr_table = wandb.Table(columns=["step", "input", "prediction", "label"],
                         log_mode="INCREMENTAL")

# Training loop
for step in range(get_num_batches()):
    # Process batch
    inputs, labels = get_training_batch(step)
    predictions = model.predict(inputs)

    # Add data to incremental table
    for inp, pred, label in zip(inputs, predictions, labels):
        incr_table.add_data(step, inp, pred, label)

    # Log the incremental update (suffix with -incr to distinguish from final table)
    run.log({"table-incr": incr_table}, step=step)

# At the end of training, create a complete immutable table with all data
# Using the default IMMUTABLE mode to preserve the complete dataset
final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data, log_mode="IMMUTABLE")
run.log({"table": final_table})

run.finish()
```

In the previous example, the `incr_table` is logged incrementally (with `log_mode="INCREMENTAL"`) during training. This allows you to log and view updates to the table as new data is processed. At the end of training, an immutable table (`final_table`) is created with all data from the incremental table. The immutable table is logged to preserve the complete dataset for further analysis and it enables you to view all rows in the W&B App. 


## Examples 

### Enriching evaluation results with MUTABLE

```python
import wandb
import numpy as np

run = wandb.init(project="mutable-logging")

# Step 1: Log initial predictions
table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
inputs, labels = load_eval_data()
raw_preds = model.predict(inputs)

for inp, label, pred in zip(inputs, labels, raw_preds):
    table.add_data(inp, label, pred)

run.log({"eval_table": table})  # Log raw predictions

# Step 2: Add confidence scores (e.g. max softmax)
confidences = np.max(raw_preds, axis=1)
table.add_column("confidence", confidences)
run.log({"eval_table": table})  # Add confidence info

# Step 3: Add post-processed predictions
# (e.g., thresholded or smoothed outputs)
post_preds = (confidences > 0.7).astype(int)
table.add_column("final_prediction", post_preds)
run.log({"eval_table": table})  # Final

run.finish()
```

### Resuming runs with INCREMENTAL tables

You can continue logging to an incremental table when resuming a run:

```python
# Start or resume a run
resumed_run = wandb.init(project="resume-incremental", id="your-run-id", resume="must")

# Create the incremental table; no need to populate with data from preivously logged table
# Increments will be continue to be added to the Table artifact.
table = wandb.Table(columns=["step", "metric"], log_mode="INCREMENTAL")

# Continue logging
for step in range(resume_step, final_step):
    metric = compute_metric(step)
    table.add_data(step, metric)
    resumed_run.log({"metrics": table}, step=step)

resumed_run.finish()
```

### Training with INCREMENTAL batch training

```python

run = wandb.init(project="batch-training-incremental")

# Create an incremental table
table = wandb.Table(columns=["step", "input", "label", "prediction"], log_mode="INCREMENTAL")

# Simulated training loop
for step in range(get_num_batches()):
    # Load batch data
    inputs, labels = get_training_batch(step)

    # Train the model on this batch
    train_model_on_batch(inputs, labels)

    # Run model inference
    predictions = predict_on_batch(inputs)

    # Add data to the table
    for input_item, label, prediction in zip(inputs, labels, predictions):
        table.add_data(step, input_item, label, prediction)

    # Log the current state of the table incrementally
    run.log({"training_table": table}, step=step)

run.finish()