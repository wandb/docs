---
description:
  Learn how to use MUTABLE and INCREMENTAL logging modes for W&B Tables
menu:
  default:
    identifier: tables-logging-modes
    parent: tables
title: 'Table Logging Modes: IMMUTABLE, MUTABLE and INCREMENTAL'
---

W&B Tables support logging modes that solve common problems when working with
table data: `IMMUTABLE`, `MUTABLE` and `INCREMENTAL`. These modes give you more
flexibility and control over how your tabular data is logged and updated during
your ML experiments.

## Understanding Table Logging Modes

Unlike other W&B media types, Tables are logged as artifacts and have different
behavior by default:

- **IMMUTABLE mode** (default): Tables can only be logged once per key, with
  subsequent log calls being no-ops.
- **MUTABLE mode**: Allows you to update an already logged table by replacing it
  entirely.
- **INCREMENTAL mode**: Allows you to add batches of new rows to a table over
  time.

## When to Use Each Mode

Choose the right logging mode based on your workflow needs:

| Mode            | Use Cases                                                                                              | Benefits                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------- |
| **IMMUTABLE**   | - Storing tabular data generated at the end of a run for further analysis                              | - Minimal overhead when logged at the end of a run<br>- All rows rendered in UI |
| **MUTABLE**     | - Adding columns to existing tables<br>- Enriching results with new information                        | - Capture Table mutations<br>- All rows rendered in UI                          |
| **INCREMENTAL** | - Long-running training jobs<br>- Processing large datasets in batches<br>- Monitoring ongoing results | - View updates on UI during training<br>- Ability to step through increments    |

## Using MUTABLE Mode

With `MUTABLE` mode, you can update a table multiple times while keeping the
same key. This is useful when you need to add columns or modify existing data.

```python
import wandb

run = wandb.init(project="mutable-table-demo")

# Create a table with MUTABLE logging mode
table = wandb.Table(columns=["input", "label", "prediction"],
                    log_mode="MUTABLE")

# Step 1: Log initial data
inputs, labels = load_eval_data()
raw_preds = model.predict(inputs)

for inp, label, pred in zip(inputs, labels, raw_preds):
    table.add_data(inp, label, pred)

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

When using `MUTABLE` mode, your table will be completely replaced each time you
log it, and the UI will display the most recent version.

## Example

### Enriching Evaluation Results

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

## Using INCREMENTAL Mode

With `INCREMENTAL` mode, you can log batches of rows to a table over time. This
is ideal for monitoring long-running jobs or when working with large datasets
that would be inefficient to log all at once.

```python
import wandb

run = wandb.init(project="incremental-table-demo")

# Create a table with INCREMENTAL logging mode
table = wandb.Table(columns=["step", "input", "label", "prediction"],
                    log_mode="INCREMENTAL")

# Training loop
for step in range(get_num_batches()):
    # Load batch data
    inputs, labels = get_training_batch(step)

    # Train and predict
    train_model_on_batch(inputs, labels)
    predictions = predict_on_batch(inputs)

    # Add batch data to table
    for input_item, label, prediction in zip(inputs, labels, predictions):
        table.add_data(step, input_item, label, prediction)

    # Log the table incrementally
    wandb.log({"training_table": table}, step=step)

run.finish()
```

When using `INCREMENTAL` mode:

- Each log call creates a new increment containing rows added since the last log

Visualization with Query Panels:

- In a project workspace, you'll see the latest increment
- In a run workspace, you can view up to the latest 100 increments
- You can use the Query Panel Stepper to navigate through increments by step

## Resuming Runs with INCREMENTAL Tables

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

## Performance Considerations

- **For very large tables**: If logging the table at multiple steps,
  `log_mode="INCREMENTAL"` will be more efficient
- **For many small increments**: Consider using `wandb.log()` with history
  instead of tables
- **Visualization**:
  - Project workspace: Only latest increment shown
  - Run workspace: Latest 100 increments shown

## Example

### Training with Incremental Batch Logging

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
```

## Combining INCREMENTAL and IMMUTABLE Logging

While INCREMENTAL logging is more efficient than logging the entire table, it
comes with UI visualization limitations. The limitation can be mitigated by
combining INCREMENTAL logging with IMMUTABLE logging at the end:

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
final_table = wandb.Table(columns=incr_table.columns, data=incr_table.data)
run.log({"table": final_table})

run.finish()
```

- During training: Log incremental updates for efficiency and real-time
  monitoring
- At completion: Log the entire dataset once as an immutable table for:
  - Complete visualization without the 100-increment limit in run workspaces
  - Full dataset access in project workspaces (not just the latest increment)
  - Better query performance for analysis after training
