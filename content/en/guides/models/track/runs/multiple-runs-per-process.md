---
description: Manage multiple runs in a single Python process using W&B’s reinit functionality
menu:
  default:
    identifier: multiple-runs
    parent: what-are-runs
title: Create and manage multiple runs in a single process
---

Manage multiple runs in a single Python process. This is useful for workflows where you want to keep a primary process active while creating short-lived secondary processes for sub-tasks. Some use cases include:

- Keeping a single “primary” run active throughout a script while spinning up short-lived “secondary” runs for evaluations or sub-tasks.  
- Orchestrating sub-experiments in a single file.  
- Logging from one “main” process to several runs that represent different tasks or time periods.

By default, W&B assumes each Python process has only one active run at a time when you call `wandb.init()`. If you call `wandb.init()` again, W&B will either return the same run or finish the old run before starting a new one, depending on the configuration. The content in this guide explains how to use `reinit` to modify the `wandb.init()` behavior to enable multiple runs in a single Python process.

{{% alert title="Requirements" %}}
To manage multiple runs in a single Python process, you must have W&B Python SDK version `v0.19.10` or newer.
{{% /alert  %}}

## `reinit` options

Use the `reinit` parameter to configure how W&B handles multiple calls to `wandb.init()`. The following table describes valid arguments and their effects:

| | Description | Creates a run? | Example use case |
|----------------|----------------|----------------| -----------------|
| `create_new` |Create a new run with `wandb.init()` without finishing existing, active runs. W&B does not automatically switch the global `wandb.run` to new runs. You must hold onto each run object yourself. See the [example]({{< relref "multiple-runs-per-process/#example-multiple-runs-in-one-process" >}}) below for details.  | Yes |  Ideal for creating and managing concurrent processes. For example, a “primary” run that remains active while you start or end “secondary” runs.|
| `finish_previous` | Finish all active runs with `run.finish()` before creating a new one run with `wandb.init()`. Default behavior for non notebook environments. | Yes | Ideal when you want to break sequential sub-processes into separate individual runs. |
| `return_previous` |  Return the most recent, unfinished run. Default behavior for notebook environments. | No | |

{{% alert  %}}
W&B does not support `create_new` mode for [W&B Integrations]({{< relref "/guides/integrations/" >}}) that assume a single global run, such as Hugging Face Trainer, Keras callbacks, and PyTorch Lightning. If you use these integrations, you should run each sub-experiment in a separate process.
{{% /alert %}}

## Specifying `reinit`

<!-- There are several ways to create and manage multiple runs in a single Python process: -->

- Use `wandb.init()` with the `reinit` argument directly:
   ```python
   import wandb
   wandb.init(reinit="<create_new|finish_previous|return_previous>")
   ```
- Use `wandb.init()` and pass a `wandb.Settings` object to the `settings` parameter. Specify `reinit` in the `Settings` object:

   ```python
   import wandb
   wandb.init(settings=wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- Use `wandb.setup()` to set the `reinit` option globally for all runs in the current process. This is useful if you want to configure the behavior once and have it apply to all subsequent `wandb.init()` calls in that process.

   ```python
   import wandb
   wandb.setup(wandb.Settings(reinit="<create_new|finish_previous|return_previous>"))
   ```

- Specify the desired value for `reinit` in the environment variable `WANDB_REINIT`. Defining an environment variable applies the `reinit` option to `wandb.init()` calls.

   ```bash
   export WANDB_REINIT="<create_new|finish_previous|return_previous>"
   ```

The following code snippet shows a high level overview how to set up W&B to create a new run each time you call `wandb.init()`:

```python
import wandb

wandb.setup(wandb.Settings(reinit="create_new"))

with wandb.init() as experiment_results_run:
   for ...:
      with wandb.init() as run:
         # The do_experiment() function logs fine-grained metrics
         # to the given run and returns result metrics that
         # you want to track separately.
         experiment_results = do_experiment(run)

         # After each experiment, log its results to a parent
         # run. Each point in the parent run's charts corresponds
         # to one experiment's results.
         experiment_results_run.log(experiment_results)
```

## Example: Concurrent processes

Suppose you want to create a primary process that remains open for the script's entire lifespan, while periodically spawning short-lived secondary processes without finishing the primary process. For example, this pattern can be useful if you want to train a model in the primary run, but compute evaluations or do other work in separate runs.

To achieve this, use `reinit="create_new"` and initialize multiple runs. For this example, suppose "Run A" is the primary process that remains open throughout the script, while "Run B1", "Run B2", are short-lived secondary runs for tasks like evaluation. 

The high level workflow might look like this:

1. Initialize the primary process Run A with `wandb.init()` and log training metrics.  
2. Initialize Run B1 (with `wandb.init()`), log data, then finish it.  
3. Log more data to Run A.  
4. Initialize Run B2, log data, then finish it.  
5. Continue logging to Run A.  
6. Finally finish Run A at the end.

The following Python code example demonstrates this workflow:

```python
import wandb

def train(name: str) -> None:
    """Perform one training iteration in its own W&B run.

    Using a 'with wandb.init()' block with `reinit="create_new"` ensures that
    this training sub-run can be created even if another run (like our primary
    tracking run) is already active.
    """
    with wandb.init(
        project="my_project",
        name=name,
        reinit="create_new"
    ) as run:
        # In a real script, you'd run your training steps inside this block.
        run.log({"train_loss": 0.42})  # Replace with your real metric(s)

def evaluate_loss_accuracy() -> (float, float):
    """Returns the current model's loss and accuracy.
    
    Replace this placeholder with your real evaluation logic.
    """
    return 0.27, 0.91  # Example metric values

# Create a 'primary' run that remains active throughout multiple train/eval steps.
with wandb.init(
    project="my_project",
    name="tracking_run",
    reinit="create_new"
) as tracking_run:
    # 1) Train once under a sub-run named 'training_1'
    train("training_1")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})

    # 2) Train again under a sub-run named 'training_2'
    train("training_2")
    loss, accuracy = evaluate_loss_accuracy()
    tracking_run.log({"eval_loss": loss, "eval_accuracy": accuracy})
    
    # The 'tracking_run' finishes automatically when this 'with' block ends.
```

Note three key points from the previous example:

1. `reinit="create_new"` creates a new run each time you call `wandb.init()`.
2. You keep references of each run. `wandb.run` does not automatically point to the new run created with `reinit="create_new"`. Store new runs in variables like `run_a`, `run_b1`, etc., and call `.log()` or `.finish()` on those objects as needed.
3. You can finish sub-runs whenever you want while keeping the primary run open until.
4. Finish your runs with `run.finish()` when you are done logging to them. This ensures that all data is uploaded and the run is properly closed.


