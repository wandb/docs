---
menu:
  default:
    identifier: code
    parent: panels
title: Save and diff code
weight: 50
---

By default, W&B only saves the latest git commit hash. You can turn on more code features to compare the code between your experiments dynamically in the UI.

Starting with `wandb` version 0.8.28, W&B can save the code from your main training file where you call `wandb.init()`. 

## Save library code

When you enable code saving, W&B saves the code from the file that called `wandb.init()`. To save additional library code, you have three options:

### Call `wandb.run.log_code(".")` after calling `wandb.init()`

```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

### Pass a settings object to `wandb.init` with `code_dir` set

```python
import wandb

wandb.init(settings=wandb.Settings(code_dir="."))
```

This captures all python source code files in the current directory and all subdirectories as an [artifact](../../../../ref/python/artifact.md). For more control over the types and locations of source code files that are saved, see the [reference docs](../../../../ref/python/run.md#log_code).

### Set code saving in the UI

In addition to setting code saving programmatically, you can also toggle this feature in your W&B account Settings. Note that this will enable code saving for all teams associated with your account.

> By default, W&B disables code saving for all teams.

1. Log in to your W&B account.
2. Go to **Settings** > **Privacy**.
3. Under **Project and content security**, toggle **Disable default code saving** on. 

## Code comparer
Compare code used in different W&B runs:

1. Select the **Add panels** button in the top right corner of the page.
2. Expand **TEXT AND CODE** dropdown and select **Code**.


{{< img src="/images/app_ui/code_comparer.png" alt="" >}}

## Jupyter session history

W&B saves the history of code executed in your Jupyter notebook session. When you call **wandb.init()** inside of Jupyter, W&B adds a hook to automatically save a Jupyter notebook containing the history of code executed in your current session. 


1. Navigate to your project workspaces that contains your code.
2. Select the **Artifacts** tab in the left navigation bar.
3. Expand the **code** artifact.
4. Select the **Files** tab.

{{< img src="/images/app_ui/jupyter_session_history.gif" alt="" >}}

This displays the cells that were run in your session along with any outputs created by calling iPython’s display method. This enables you to see exactly what code was run within Jupyter in a given run. When possible W&B also saves the most recent version of the notebook which you would find in the code directory as well.

{{< img src="/images/app_ui/jupyter_session_history_display.png" alt="" >}}