---
description: Learn to manage multiple runs in a single Python process using W&B’s reinit functionality
menu:
  default:
    identifier: multiple-runs
    parent: what-are-runs
title: Multiple Runs in One Process
---

# Simultaneous runs using reinit="create_new"

**Quick Reference**  
- **`return_previous`**: Return the existing run; no new run is created. If there are already multiple runs, this returns the most recently created one that isn't finished.  
- **`finish_previous`**: Finish all previous runs, then create a new one.
- **`create_new`**: Create a new run without finishing the old one— great for parallel or interleaved runs. Not supported for use with third-party integrations.

---

By default, W&B assumes each Python process has only **one active run** at a time. Calling `wandb.init()` again, in the default scenario, will either return the same run or finish the old run before starting a new one—depending on the configuration. However, there are workflows where you might want **multiple runs** in the same Python process:

- Keeping a single “primary” run active throughout a script while spinning up short-lived “secondary” runs for evaluations or sub-tasks.  
- Orchestrating sub-experiments in a single file.  
- Logging from one “master” process to several runs that represent different tasks or time periods.

W&B provides a **`reinit`** setting to control how `wandb.init()` behaves when there’s an existing active run. Below, we detail each `reinit` option and show how to maintain multiple concurrent runs—finishing them at different times without interference.

---

## Overview of `reinit` Options
Set the reinit setting in any of the following ways:
- Passing it as the reinit argument to wandb.init() or via the settings argument
- Passing it via settings to wandb.setup() (applies to all wandb.init() calls)
- Setting the `WANDB_REINIT` environment variable (applies to all wandb.init() calls)


1. **`return_previous`**  
   - If another run is active, return *that same* run.  
   - *No new run is created.* This is effectively a no-op and is the legacy default when not running in a Jupyter/IPython notebook.

2. **`finish_previous`**  
   - *Finish* (calls `run.finish()`) on any active run(s) before creating a new one.  
   - Handy if you want a clear break between sequential sub-runs.
   - This is the default when running in a Jupyter/IPython notebook.
3. **`create_new`**  
   - Always create a **completely new run**, even if one is already active.  
   - Does **not** automatically switch the global `wandb.run` to the new run. You must hold onto each run object yourself.  
   - Ideal for truly concurrent runs (e.g., a “primary” run that remains active while you start/end “secondary” runs).  
   - **Caution**: Many W&B integrations (e.g., Hugging Face Trainer, Keras callbacks, PyTorch Lightning) assume a single global run. Currently, integrations do not support `create_new` mode.

---

## Example: Maintaining a Primary (Long-Lived) Run + Sub-Runs

Suppose you want to run a **primary task** (like model training) in “Run A,” which stays open for the script’s entire lifespan. Meanwhile, you periodically spawn **short-lived “secondary” runs** (Run B1, B2, etc.) for tasks like evaluation—**without** finishing Run A.

### Workflow Overview

1. **Initialize Run A** and log training metrics.  
2. **Initialize Run B1** (evaluation), log data, then finish it.  
3. **Log more data to Run A**.  
4. **Initialize Run B2**, log data, then finish it.  
5. **Continue logging to Run A**.  
6. **Finally finish Run A** at the end.

### Sample Code

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
        # Pretend there's a training loop here logging 'train_loss'.
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

---

### Key Takeaways

1. **`reinit="create_new"`**  
   Creates a new run each time you call `wandb.init()`.

2. **Keep references to each run**  
   Since `wandb.run` won’t automatically point to the newest run in `create_new` mode, store them in variables like `run_a`, `run_b1`, etc., and call `.log()` or `.finish()` on those objects.

3. **Order of finishing**  
   You can finish sub-runs whenever you want while keeping the primary run open until the very end.

4.  **Always finish your runs**  
    If you don’t explicitly call `run.finish()`, your script exit may be slower, or your data may not be fully uploaded. 

---

### Typical Use Cases

1. **Long-Lived Master Run + Short Sub-Task Runs**  
   Keep a main “tracking” or “training” run open for the entire script while spinning up quick, short-lived runs (e.g., evaluations or diagnostics) that start and finish independently. This pattern gives you a continuous overview (the master run) plus separate details for each sub-task.

2. **Multiple Experiments in One Process**  
   If you have a single script that launches several experiments in sequence (but you need each experiment to have its own distinct W&B run), using `reinit="create_new"` lets you cleanly separate each experiment’s logs.

3. **A Central Logging Process**  
   When distributed workers do the heavy lifting elsewhere, you might gather all results in a single Python process. `reinit="create_new"` ensures each worker or sub-task is tracked separately, even though the logging code runs in the same environment.

--- 

## When to Consider a Single Run Per Active Process

While `reinit="create_new"` enables some advanced workflows, it also adds complexity. You might **avoid** multiple active runs in a single process if:

1. **You Rely on Integrations That Assume a Single Run**  
   - Popular frameworks (e.g., PyTorch Lightning, Keras, Hugging Face Transformers) often expect exactly one global `wandb.run` at a time. 
   - These often attach to whichever run is active in the global scope. Creating multiple runs concurrently can lead to errors or unexpected behavior.  
   - If you rely on these integrations, you should run each sub-experiment in a **separate process**.

2. **Simplicity Is a Priority**  
   - Managing multiple run objects within one script can be more difficult to debug. If your experiments are straightforward, **one run per process** is often the easiest approach.

3. **You Don’t Need Concurrent Logging**  
   - If your training and evaluation can be done sequentially, a single run (or finishing one run before starting another) is simpler and less error-prone.


