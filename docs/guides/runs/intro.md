---
slug: /guides/runs
description:  Learn about the basic building block of W&B, Runs.
---
# Runs

A single unit of computation logged by W&B is called a *Run*. 

Consider a W&B Run as an atomic element of your whole project. You should create and initiate a new Run if you change a hyperparameter, use a different model, create a [W&B Artifact](../artifacts/intro.md) and so on.

For example, in a [W&B Sweep](../sweeps/intro.md), W&B explores a hyperparameter search and explores the space of possible models. Each new hyperparameter combination is implemented as a W&B Run. 

Use W&B Runs for tasks such as:

* Each time you train a model.
* Log data or a model as a [W&B Artifact](../artifacts/intro.md).
* [Download a W&B Artifact](../artifacts/download-and-use-an-artifact.md).


Anything you log with `wandb.log` is recorded in that Run.  For more information on how log objects in W&B, see [Log Media and Objects](../track/log/intro.md).

View Runs within a project within your Project's [Workspace](#view-runs). 

## Create a Run

Create a W&B Run with [`wandb.init()`](../../ref/python/init.md):

```python
import wandb

run = wandb.init(project='my-project-name')
```

Optionally provide the name of a project for the `project` field. We recommend you specify a project name when you create a Run object. W&B creates a new project if a project does not already exist with the name you provide.  Projects help organize experiments, runs, artifacts, and more in one convenient location called a *Project Workspace*. A Project's Workspace gives you a personal sandbox to compare runs.

:::info
If a project is not specified, the W&B Run is stored in a project called "Uncategorized".
:::

There is only ever at most one active [`wandb.Run`](../../ref/python/run.md) in any process,
and it is accessible as `wandb.run`:

```python
import wandb

assert wandb.run is None

wandb.init()

assert wandb.run is not None
```


You need to finish a Run that has not completed in order to start one or more Runs in the same notebook or script. 




## End a Run
W&B automatically calls [`wandb.finish`](../../ref/python/finish.md) to finalize and cleanup a run. However, if you call [`wandb.init`](../../ref/python/init.md) from a child process, you must explicitly call `wandb.finish` at the end of the child process. 

:::note
The wandb.finish API is automatically called when your script exits.
:::

You can end a Run manually with the [`wandb.finish`](../../ref/python/finish.md) API or end a Run using a `with` statement. The following code example demonstrates how to end a Run from a `with` Python statement:

```python
import wandb

wandb.init()
wandb.finish()

assert wandb.run is None

with wandb.init() as run:
 pass # log data here

assert wandb.run is None
```


## View all Runs in a Project
View Runs associated to a project with the W&B App UI. Navigate to the W&B App and search for the name of your project. 

In the following example we search for a project called "my-first-run":

![](/images/runs/search_run_name_landing_page.png)

Select the project. This will redirect you to that project's Workspace. A Project's Workspace gives you a personal sandbox to compare runs. Use projects to organize models that can be compared, working on the same problem with different architectures, hyperparameters, datasets, preprocessing and so on.

Within your project's workspace, you will see a table labeled **Runs**. This table lists all the Runs that are in your project. In other words, these runs were provided a `project` argument when it was created.

The following image demonstrates a project workspace called "sweep-demo":

![Example project workspace called 'sweep-demo'](/images/app_ui/workspace_tab_example.png)

The **Runs Sidebar** lists of all the runs in your project. Hover your mouse over a single Run to modify or view the following:

* **Kebob menu**: Use this kebob menu to rename a Run, delete a Run, or stop an active Run.
* **Visibility icon**: Select the eye icon to hide specific run.
* **Color**: change the run color to another one of our presets or a custom color.
* **Search**: search runs by name. This also filters visible runs in the plots.
* **Filter**: use the sidebar filter to narrow down the set of runs visible.
* **Group**: select a config column to dynamically group your runs, for example by architecture. Grouping makes plots show up with a line along the mean value, and a shaded region for the variance of points on the graph.
* **Sort**: pick a value to sort your runs by, for example runs with the lowest loss or highest accuracy. Sorting will affect which runs show up on the graphs.
* **Expand button**: expand the sidebar into the full table
* **Run count**: the number in parentheses at the top is the total number of runs in the project. The number (N visualized) is the number of runs that have the eye turned on and are available to be visualized in each plot. In the example below, the graphs are only showing the first 10 of 183 runs. Edit a graph to increase the max number of runs visible.

For more information about how to organize multiple Runs in a project, see the [Runs Table](../app/features/runs-table.md) documentation. 

For a live example of a Project's Workspace, [see this example project](https://app.wandb.ai/example-team/sweep-demo). 



<!-- ### Search runs

Search for a specific run by name in the sidebar. You can use regex to filter down your visible runs. The search box affects which runs are shown on the graph. Here's an example:

![](/images/app_ui/project_page_search_for_runs.gif)

### Filter runs

### Organize runs -->




## Investigate a specific Run in a Project

Use the run page to explore detailed information about a specific Run. 

1. Navigate to your project and select a specific Run from the **Runs Sidebar**.
2. Next, select the **Overview Tab** icon. 

The following image demonstrates information about a Run called "sparkling-glade-2":

![W&B Dashboard run overview tab](/images/app_ui/wandb_run_overview_page.png)

The **Overview Tab** will show the following information about the Run you selected:

* Run name: The name of the Run.
* Description: A description of the Run that you provided. This field is left initially blank if no description was specified when you create the Run. You can optionally provide a description for the Run with the W&B App UI or programmatically. 
* Privacy: Privacy settings of the Run. You can set it to either **Private** or **Public**. 
    * **Private**: (Default) Only you can view and contribute.
    * **Public**: Anyone can view.
* Tags: (list, optional) A list of strings. Tags are useful for organizing runs together, or applying temporary labels like "baseline" or "production".
* Author: The W&B username that created the Run.
* Run state: The state of the Run:
  * **finished**: script ended and fully synced data, or called `wandb.finish()`
  * **failed**: script ended with a non-zero exit status
  * **crashed**: script stopped sending heartbeats in the internal process, which can happen if the machine crashes
  * **running**: script is still running and has recently sent a heartbeat
* Start time: The timestamp when the Run started.
* Duration: How long, in seconds, the Run took to **finish**, **fail**, or **crash**.
* Host name: Where the Run was launched. The name of your machine is displayed if you launched the Run locally on your machine. 
* Operating system: The operating system used for the Run.
* Python version: The Python version used for the Run.
* Python executable: The command that started the Run.
* System Hardware: The hardware of the 
* W&B CLI version: The W&B ClI version installed on the machine that hosted the Run command.

<!-- :::info
The Python details are private, even if you make the page itself public. 
::: -->


Below the overview section, you will additionally find information about: 

* **Artifact Outputs**: Artifact outputs produced by the Run.
* **Config**: List of config parameters saved with [`wandb.config`](../../guides/track/config.md).
* **Summary**: List of summary parameters saved with [`wandb.log()`](../../guides/track/log/intro.md). By default, this value is set to the last value logged.






