---
title: Log tables
weight: 1
---

Visualize and log tabular data with W&B Tables. A table is a two-dimensional grid of data where each column has a single type of data. Each row represents one or more data points logged to a W&B [run]({{< relref "/guides/models/track/runs/" >}}). Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types.

<!-- Tables are a specialized [data type]({{< relref "/ref/python/data-types/" >}}) in W&B that are logged internally in W&B as an [artifact]({{< relref "/guides/core/artifacts/" >}}) objects. -->

You [create and log tables]({{< relref "#create-and-log-a-new-table" >}}) using the W&B Python SDK. You can make a table [immutable, mutable, or incremental]({{< relref "#table-logging-modes" >}}). The mode you set determines how if and how you can modify the table after it is created.

## Create a table

Log a table to W&B using the `wandb.Table` class. 

1. First, create a new run with `wandb.init()`. 
2. Next, create a table object with `wandb.Table`. Specify the columns and data for the table for the `columns` and `data` parameters, respectively. 
3. Finally, log the table to W&B with `run.log()`.

The following example shows how to create and log a table with two columns, `a` and `b`, and two rows of data, `["a1", "b1"]` and `["a2", "b2"]`:

```python
import wandb

with wandb.init(project="table-test") as run:
    my_table = wandb.Table(
        columns=["a", "b"], data=[["a1", "b1"], ["a2", "b2"]],
        log_mode="IMMUTABLE"
    )
    run.log({"Table Name": my_table})
```

## Logging modes

There are three table modes. The mode you set when you create a table determines how if and how you can modify the table after it is created. The three modes are:

- `IMMUTABLE` mode (default): Once a table is logged, it cannot be modified.
- `MUTABLE` mode: Update an existing table by replacing the existing table with a new one.
- `INCREMENTAL` mode: Add batches of new rows to a table iteratively.

The following table describes the differences between the three modes and common use cases for each:

| Mode  | Use Cases  | Benefits  |
| ----- | ---------- | ----------|
| `IMMUTABLE`   | - Storing tabular data generated at the end of a run for further analysis                              | - Minimal overhead when logged at the end of a run<br>- All rows rendered in UI |
| `MUTABLE`     | - Adding columns to existing tables<br>- Enriching results with new information                        | - Capture Table mutations<br>- All rows rendered in UI                          |
| `INCREMENTAL` | - Long-running training jobs<br>- Processing large datasets in batches<br>- Monitoring ongoing results | - View updates on UI during training<br>- Ability to step through increments    |

The next sections show example code snippets for each mode along with considerations for when to use each mode.

### MUTABLE mode

`MUTABLE` mode updates an existing table by replacing the existing table with a new one. `MUTABLE` mode is useful when you want to add new columns and rows to an existing table in a non iterative process. If you want to add new rows incrementally like in a training loop, consider using [`INCREMENTAL` mode]({{< relref "#INCREMENTAL-mode" >}}) instead.

The following example shows how to create a table in `MUTABLE` mode, log it, and then add new columns to it.

{{% alert %}}
The following example uses a placeholder function `load_eval_data()` to load data and a placeholder function `model.predict()` to make predictions. You will need to replace these with your own data loading and prediction functions.
{{% /alert %}}

```python
import wandb

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

# Step 2: Add confidence scores
confidences = compute_confidences(raw_preds)
table.add_column("confidence", confidences)
wandb.log({"eval_table": table})  # Updates the table with new column

# Step 3: Add post-processed predictions
post_preds = postprocess_predictions(raw_preds, confidences)
table.add_column("final_prediction", post_preds)
wandb.log({"eval_table": table})  # Final update with another column

run.finish()
```

Each call to `wandb.log()` replaces the existing table with a new one. In the previous example, the table is logged three times: once with the initial data, once with the confidence scores, and once with the final predictions.


### INCREMENTAL mode

Incremental logging is generally more computationally efficient than logging a new table each time (`log_mode=MUTABLE`). It is especially useful for long-running training jobs or when processing large datasets in batches.

<!-- - **For many small increments**: Consider using `wandb.log()` with history
  instead of tables -->

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

Each call to `wandb.log()` appends new rows to an existing table. In the previous code example, the table is logged once per training step. The `step` parameter is used to specify the training step for each log call. The table is updated with new rows for each batch of data processed in the training loop. 

{{% alert %}}
Run workspaces in the W&B App have a limit of 100 increments for tables. If you log more than 100 increments, only the most recent 100 are shown in the run workspace. You can use the step slider to view through the increments to see how the table changes over time. 
{{% /alert %}}

In some cases, the efficiency of rendering a table in the W&B App may be impacted by the number of increments you log. In this case, consider combining `INCREMENTAL` and `IMMUTABLE` logging modes to reduce the number of increments shown in the W&B App.

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

In the previous example, the `incr_table` is logged incrementally (with `log_mode="INCREMENTAL"`) during training. Each time a new batch is processed, the table is updated with new rows. This allows you to monitor the training process in real-time and visualize the incremental updates in the W&B App.

At the end of training, a complete immutable table is created with all data from the incremental table. This allows you to preserve the complete dataset while still benefiting from the efficiency of incremental logging during training.

## Examples 

### Enriching evaluation results with MUTABLE

```python
run = wandb.init(project="mutable-logging")

# Step 1: Log initial predictions
table = wandb.Table(columns=["input", "label", "prediction"], log_mode="MUTABLE")
inputs, labels = load_eval_data()
raw_preds = model.predict(inputs)

for inp, label, pred in zip(inputs, labels, raw_preds):
    table.add_data(inp, label, pred)

run.log({"eval_table": table})  # Log raw predictions

# Step 2: Add confidence scores (e.g. max softmax)
confidences = compute_confidences(raw_preds)
table.add_column("confidence", confidences)
run.log({"eval_table": table})  # Add confidence info

# Step 3: Add post-processed predictions
# (e.g., thresholded or smoothed outputs)
post_preds = postprocess_predictions(raw_preds, confidences)
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