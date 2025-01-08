---
description: Learn about the basic building block of W&B, Runs.
menu:
  default:
    identifier: what-are-runs
    parent: experiments
title: What are runs?
weight: 5
url: guides/runs
cascade:
- url: guides/runs/:filename
---


A *run* is a single unit of computation logged by W&B. You can think of a W&B run as an atomic element of your whole project. In other words, each run is a record of a specific computation, such as training a model and logging the results, hyperparameter sweeps, and so forth.

Common patterns for initiating a run include, but are not limited to: 

* Training a model
* Changing a hyperparameter and conducting a new experiment
* Conducting a new machine learning experiment with a different model
* Logging data or a model as a [W&B Artifact](../artifacts/intro.md)
* [Downloading a W&B Artifact](../artifacts/download-and-use-an-artifact.md)


W&B stores runs that you create into [*projects*](../track/project-page.md). You can view runs and their properties within the run's project workspace on the W&B App UI. You can also programmatically access run properties with the [`wandb.Api.Run`](../../ref/python/public-api/run.md) object.

Anything you log with `run.log` is recorded in that run. Consider the proceeding code snippet.

```python
import wandb

run = wandb.init(entity="nico", project="awesome-project")
run.log({"accuracy": 0.9, "loss": 0.1})
```

The first line imports the W&B Python SDK. The second line initializes a run in the project `awesome-project` under the entity `nico`. The third line logs the accuracy and loss of the model to that run.

Within the terminal, W&B returns:

```bash
wandb: Syncing run earnest-sunset-1
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb:                                                                                
wandb: 
wandb: Run history:
wandb: accuracy ▁
wandb:     loss ▁
wandb: 
wandb: Run summary:
wandb: accuracy 0.9
wandb:     loss 0.5
wandb: 
wandb: 🚀 View run earnest-sunset-1 at: https://wandb.ai/nico/awesome-project/runs/1jx1ud12
wandb: ⭐️ View project at: https://wandb.ai/nico/awesome-project
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20241105_111006-1jx1ud12/logs
```

The URL W&B returns in the terminal to redirects you to the run's workspace in the W&B App UI. Note that the panels generated in the workspace corresponds to the single point.

{{< img src="/images/runs/single-run-call.png" alt="" >}}

Logging a metrics at a single point of time might not be that useful. A more realistic example in the case of training discriminative models is to log metrics at regular intervals. For example, consider the proceeding code snippet:

```python
epochs = 10
lr = 0.01

run = wandb.init(
    entity="nico",
    project="awesome-project",
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)

offset = random.random() / 5

# simulating a training run
for epoch in range(epochs):
    acc = 1 - 2**-epoch - random.random() / (epoch + 1) - offset
    loss = 2**-epoch + random.random() / (epoch + 1) + offset
    print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
    run.log({"accuracy": acc, "loss": loss})
```

This returns the following output:

```bash
wandb: Syncing run jolly-haze-4
wandb: ⭐️ View project at https://wandb.ai/nico/awesome-project
wandb: 🚀 View run at https://wandb.ai/nico/awesome-project/runs/pdo5110r
lr: 0.01
epoch=0, accuracy=-0.10070974957523078, loss=1.985328507123956
epoch=1, accuracy=0.2884687745057535, loss=0.7374362314407752
epoch=2, accuracy=0.7347387967382066, loss=0.4402409835486663
epoch=3, accuracy=0.7667969248039795, loss=0.26176963846423457
epoch=4, accuracy=0.7446848791003173, loss=0.24808611724405083
epoch=5, accuracy=0.8035095836268268, loss=0.16169791827329466
epoch=6, accuracy=0.861349032371624, loss=0.03432578493587426
epoch=7, accuracy=0.8794926436276016, loss=0.10331872172219471
epoch=8, accuracy=0.9424839917077272, loss=0.07767793473500445
epoch=9, accuracy=0.9584880427028566, loss=0.10531971149250456
wandb: 🚀 View run jolly-haze-4 at: https://wandb.ai/nico/awesome-project/runs/pdo5110r
wandb: Find logs at: wandb/run-20241105_111816-pdo5110r/logs
```

The training script calls `run.log` 10 times. Each time the script calls `run.log`, W&B logs the accuracy and loss for that epoch. Selecting the URL that W&B prints from the preceding output, directs you to the run's workspace in the W&B App UI.

Note that W&B captures the simulated training loop within a single run called `jolly-haze-4`. This is because the script calls `wandb.init` method only once. 

{{< img src="/images/runs/run_log_example_2.png" alt="" >}}

As another example, during a [sweep](../sweeps/intro.md), W&B explores a hyperparameter search space that you specify. W&B implements each new hyperparameter combination that the sweep creates as a unique run.


## Initialize a run

Initialize a W&B run with [`wandb.init()`](../../ref/python/init.md). The proceeding code snippet shows how to import the W&B Python SDK and initialize a run. 

Ensure to replace values enclosed in angle brackets (`< >`) with your own values:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
```

When you initialize a run, W&B logs your run to the project you specify for the project field (`wandb.init(project="<project>"`). W&B creates a new project if the project does not already exist. If the project already exists, W&B stores the run in that project.

{{% alert %}}
If you do not specify a project name, W&B stores the run in a project called `Uncategorized`.
{{% /alert %}}

Each run in W&B has a [unique identifier known as a *run ID*](#unique-run-identifiers). [You can specify a unique ID](#unique-run-identifiers) or let [W&B randomly generate one for you](#autogenerated-run-ids).

Each run also has a human-readable,[ non-unique identifier known as a *run name*](#name-your-run). You can specify a name for your run or let W&B randomly generate one for you.

For example, consider the proceeding code snippet: 

```python title="basic.py"
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
```
The code snippet produces the proceeding output:

```bash
🚀 View run exalted-darkness-6 at: 
https://wandb.ai/nico/awesome-project/runs/pgbn9y21
Find logs at: wandb/run-20241106_090747-pgbn9y21/logs
```

Since the preceding code did not specify an argument for the id parameter, W&B creates a unique run ID. Where `nico` is the entity that logged the run, `awesome-project` is the name of the project the run is logged to, `exalted-darkness-6` is the name of the run, and `pgbn9y21` is the run ID.

{{% alert title="Notebook users" %}}
Specify `run.finish()` at the end of your run to mark the run finished. This helps ensure that the run is properly logged to your project and does not continue in the background.

```python title="notebook.ipynb"
import wandb

run = wandb.init(entity="<entity>", project="<project>")
# Training code, logging, and so forth
run.finish()
```
{{% /alert %}}

Each run has a state that describes the current status of the run. See [Run states](#run-states) for a full list of possible run states.

## Run states
The proceeding table describes the possible states a run can be in: 

| State | Description |
| ----- | ----- |
| Finished| Run ended and fully synced data, or called `wandb.finish()` |
| Failed | Run ended with a non-zero exit status | 
| Crashed | Run stopped sending heartbeats in the internal process, which can happen if the machine crashes | 
| Running | Run is still running and has recently sent a heartbeat  |


## Unique run identifiers

Run IDs are unique identifiers for runs. By default, [W&B generates a random and unique run ID for you](#autogenerated-run-ids) when you initialize a new run. You can also [specify your own unique run ID](#custom-run-ids) when you initialize a run. 

### Autogenerated run IDs

If you do not specify a run ID when you initialize a run, W&B generates a random run ID for you. You can find the unique ID of a run in the W&B App UI.

1. Navigate to the W&B App UI at [https://wandb.ai/home](https://wandb.ai/home).
2. Navigate to the W&B project you specified when you initialized the run.
3. Within your project's workspace, select the **Runs** tab.
4. Select the **Overview** tab.

W&B displays the unique run ID in the **Run path** field. The run path consists of the name of your team, the name of the project, and the run ID. The unique ID is the last part of the run path.

For example, in the proceeding image, the unique run ID is `9mxi1arc`:

{{< img src="/images/runs/unique-run-id.png" alt="" >}}


### Custom run IDs
You can specify your own run ID by passing the `id` parameter to the [`wandb.init`](../../ref/python/init.md) method. 

```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", id="<run-id>")
```

You can use a run's unique ID to directly navigate to the run's overview page in the W&B App UI. The proceeding cell shows the URL path for a specific run:

```text title="W&B App URL for a specific run"
https://wandb.ai/<entity>/<project>/<run-id>
```

Where values enclosed in angle brackets (`< >`) are placeholders for the actual values of the entity, project, and run ID.

## Name your run 
The name of a run is a human-readable, non-unique identifier. 

By default, W&B generates a random run name when you initialize a new run. The name of a run appears within your project's workspace and at the top of the [run's overview page](#overview-tab).

{{% alert %}}
Use run names as a way to quickly identify a run in your project workspace.
{{% /alert %}}

You can specify a name for your run by passing the `name` parameter to the [`wandb.init`](../../ref/python/init.md) method. 


```python 
import wandb

run = wandb.init(entity="<project>", project="<project>", name="<run-name>")
```

## Add a note to a run
Notes that you add to a specific run appear on the run page in the **Overview** tab and in the table of runs on the project page.

1. Navigate to your W&B project
2. Select the **Workspace** tab from the project sidebar
3. Select the run you want to add a note to from the run selector
4. Choose the **Overview** tab
5. Select the pencil icon next to the **Description** field and add your notes


## Stop a run
Stop a run from the W&B App or programmatically.

{{< tabpane text=true >}}
  {{% tab header="Programmatically" %}}
1. Navigate to the terminal or code editor where you initialized the run.
2. Press `Ctrl+D` to stop the run.

For example, following the preceding instructions, your terminal might looks similar to the following: 

```bash
KeyboardInterrupt
wandb: 🚀 View run legendary-meadow-2 at: https://wandb.ai/nico/history-blaster-4/runs/o8sdbztv
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20241106_095857-o8sdbztv/logs
```

Navigate to the W&B App UI to confirm the run is no longer active:

1. Navigate to the project that your run was logging to.
2. Select the name of the run. 
  {{% alert %}}
  You can find the name of the run that you stop from the output of your terminal or code editor. For example, in the preceding example, the name of the run is `legendary-meadow-2`.
  {{% /alert %}}
3. Choose the **Overview** tab from the project sidebar.

Next to the **State** field, the run's state changes from `running` to `Killed`.

{{< img src="/images/runs/stop-run-terminal.png" alt="" >}}  
  {{% /tab %}}
  {{% tab header="W&B App" %}}

1. Navigate to the project that your run is logging to.
2. Select the run you want to stop within the run selector.
3. Choose the **Overview** tab from the project sidebar.
4. Select the top button next to the **State** field.
{{< img src="/images/runs/stop-run-manual.png" alt="" >}}

Next to the **State** field, the run's state changes from `running` to `Killed`.

{{< img src="/images/runs/stop-run-manual-status.png" alt="" >}}  
  {{% /tab %}}
{{< /tabpane >}}

See [State fields](#run-states) for a full list of possible run states.

## View logged runs

View a information about a specific run such as the state of the run, artifacts logged to the run, log files recorded during the run, and more. 

{{< img src="/images/runs/demo-project.gif" alt="" >}}

To view a specific run:

1. Navigate to the W&B App UI at [https://wandb.ai/home](https://wandb.ai/home).
2. Navigate to the W&B project you specified when you initialized the run.
3. Within the project sidebar, select the **Workspace** tab.
4. Within the run selector, select the run you want to view.

Note that the URL path of a specific run has the proceeding format:

```text
https://wandb.ai/<team-name>/<project-name>/runs/<run-id>
```

Where values enclosed in angle brackets (`< >`) are placeholders for the actual values of the team name, project name, and run ID.

### Overview tab
Use the **Overview** tab to learn about specific run information in a project, such as:

* **Author**: The W&B entity that creates the run.
* **Command**: The command that initializes the run.
* **Description**: A description of the run that you provided. This field is empty if you do not specify a description when you create the run. You can add a description to a run with the W&B App UI or programmatically with the Python SDK.
* **Duration**: The amount of time the run is actively computing or logging data, excluding any pauses or waiting.
* **Git repository**: The git repository associated with the run. You must [enable git](../app/settings-page/user-settings.md#personal-github-integration) to view this field.
* **Host name**: Where W&B computes the run. W&B displays the name of your machine if you initialize the run locally on your machine.
* **Name**: The name of the run.
* **OS**: Operating system that initializes the run.
* **Python executable**: The command that starts the run.
* **Python version**: Specifies the Python version that creates the run.
* **Run path**: Identifies the unique run identifier in the form `entity/project/run-ID`.
* **Runtime**: Measures the total time from the start to the end of the run. It’s the wall-clock time for the run. Runtime includes any time where the run is paused or waiting for resources, while duration does not.
* **Start time**: The timestamp when you initialize the run.
* **State**: The [state of the run](#run-states).
* **System hardware**: The hardware W&B uses to compute the run.
* **Tags**: A list of strings. Tags are useful for organizing related runs together or applying temporary labels like `baseline` or `production`.
* **W&B CLI version**: The W&B CLI version installed on the machine that hosted the run command.
<!-- * **Git state**: -->

W&B stores the proceeding information below the overview section:

* **Artifact Outputs**: Artifact outputs produced by the run.
* **Config**: List of config parameters saved with [`wandb.config`](../../guides/track/config.md).
* **Summary**: List of summary parameters saved with [`wandb.log()`](../../guides/track/log/intro.md). By default, W&B sets this value to the last value logged. 

{{< img src="/images/app_ui/wandb_run_overview_page.png" alt="W&B Dashboard run overview tab" >}}

View an example project overview [here](https://wandb.ai/stacey/deep-drive/overview).

### Workspace tab
Use the Workspace tab to view, search, group, and arrange visualizations such as autogenerated and custom plots, system metrics, and more. 

{{< img src="/images/app_ui/wandb-run-page-workspace-tab.png" alt="" >}}

View an example project workspace [here](https://wandb.ai/stacey/deep-drive/workspace?nw=nwuserstacey)

### System tab
The **System tab** shows system metrics tracked for a specific run such as CPU utilization, system memory, disk I/O, network traffic, GPU utilization and more.

For a full list of system metrics W&B tracks, see [System metrics](../app/features/system-metrics.md).

{{< img src="/images/app_ui/wandb_system_utilization.png" alt="" >}}

View an example system tab [here](https://wandb.ai/stacey/deep-drive/runs/ki2biuqy/system?workspace=user-carey).


### Logs tab
The **Log tab** shows output printed on the command line such as the standard output (`stdout`) and standard error (`stderr`). 

Choose the **Download** button in the upper right hand corner to download the log file.

{{< img src="/images/app_ui/wandb_run_page_log_tab.png" alt="" >}}

View an example logs tab [here](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/logs).

### Files tab
Use the **Files tab** to view files associated with a specific run such as model checkpoints, validation set examples, and more

{{< img src="/images/app_ui/wandb_run_page_files_tab.png" alt="" >}}

View an example files tab [here](https://app.wandb.ai/stacey/deep-drive/runs/pr0os44x/files/media/images).

### Artifacts tab
The **Artifacts** tab lists the input and output [artifacts](../artifacts/intro.md) for the specified run.

{{< img src="/images/app_ui/artifacts_tab.png" alt="" >}}

View an example artifacts tab [here](https://wandb.ai/stacey/artifact_july_demo/runs/2cslp2rt/artifacts).


## Delete runs

Delete one or more runs from a project with the W&B App.

1. Navigate to the project that contains the runs you want to delete.
2. Select the **Runs** tab from the project sidebar.
3. Select the checkbox next to the runs you want to delete.
4. Choose the **Delete** button (trash can icon) above the table.
5. From the modal that appears, choose **Delete**.


{{% alert %}}
For projects that contain a large number of runs, you can use either the search bar to filter runs you want to delete using Regex or the filter button to filter runs based on their status, tags, or other properties. 
{{% /alert %}}

<!-- ### Search runs

Search for a specific run by name in the sidebar. You can use regex to filter down your visible runs. The search box affects which runs are shown on the graph. Here's an example:

### Filter runs

### Organize runs -->