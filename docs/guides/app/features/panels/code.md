---
displayed_sidebar: default
title: Save and diff code
---

By default, W&B only saves the latest git commit hash. You can turn on more code features to compare the code between your experiments dynamically in the UI.

Starting with `wandb` version 0.8.28, W&B can save the code from your main training file where you call `wandb.init()`. 



## Save library code

When code saving is enabled, W&B will save the code from the file that called `wandb.init()`. To save additional library code, you have two options:

* Call `wandb.run.log_code(".")` after calling `wandb.init()`
```python
import wandb

wandb.init()
wandb.run.log_code(".")
```

* Pass a settings object to `wandb.init` with `code_dir` set:
```python
import wandb
wandb.init(settings=wandb.Settings(code_dir="."))
```

This will capture all python source code files in the current directory and all subdirectories as an [artifact](../../../../ref/python/artifact.md). For more control over the types and locations of source code files that are saved, see the [reference docs](../../../../ref/python/run.md#log_code).

## Code comparer
Compare code used in different W&B runs:

1. Select the **Add panels** button in the top right corner of the page.
2. Expand **TEXT AND CODE** dropdown and select **Code**.


![](/images/app_ui/code_comparer.png)

## Jupyter session history

W&B saves the history of code executed in your Jupyter notebook session. When you call **wandb.init()** inside of Jupyter, W&B adds a hook to automatically save a Jupyter notebook containing the history of code executed in your current session. 


1. Navigate to your project workspaces that contains your code.
2. Select the **Artifacts** tab in the left navigation bar.
3. Expand the **code** artifact.
4. Select the **Files** tab.

![](/images/app_ui/jupyter_session_history.gif)

This displays the cells that were run in your session along with any outputs created by calling iPython’s display method. This enables you to see exactly what code was run within Jupyter in a given run. When possible W&B also saves the most recent version of the notebook which you would find in the code directory as well.

![](/images/app_ui/jupyter_session_history_display.png)

