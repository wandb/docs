---
displayed_sidebar: default
title: Create a run
---


Create a W&B run with [`wandb.init()`](../../ref/python/init.md):

```python
import wandb

run = wandb.init()
```

W&B recommends that you specify a project name and a W&B entity when you create a new run. W&B creates a new project (if the project does not already exist) within the W&B entity you provide. If the project already exists, W&B stores the run in that project.

:::info
If you do not specify a project name, W&B stores the run in a project called "Uncategorized".
:::

For example, the following code snippet initializes a run that is stored in a project called `model_registry_example` that is scoped within a `wandbee` entity:

```python
import wandb

run = wandb.init(entity="wandbee", \
        project="model_registry_example")
```

W&B prints the name of the run that is created along with a URL path to find out more information about that specific run. 

For example, the code snippet above produces this output:
![](/images/runs/run_example.png)


Use run names and run IDs to quickly find your experiments in your project


## Name your run 

The name of a run is a human-readable, non-unique identifier. You can use the name of a run to quickly identify the purpose of the run within the App UI.

By default, W&B generates a random name and run ID when you initialize a new run. The name of a run appears within your project's workspace and at the top of the [run's overview page](./view-runs.md#overview-tab).

You can specify a name for your run by passing the `name` parameter to the [`wandb.init`](../../ref/python/init.md) method. 


```python 
import wandb

run = wandb.init(
    entity="<project>", 
    project="<project>", 
    name="<run-name>"
)
```

:::tip
W&B suggests that you specify a project name when you initialize a run. If a project is not specified, W&B stores runs in a project called "Uncategorized".
:::

## Provide a unique run ID 

Run IDs are unique identifiers for runs. W&B generates a random run ID when you initialize a new run. You can specify a run ID by passing the `id` parameter to the [`wandb.init`](../../ref/python/init.md) method. 

```python 
import wandb

run = wandb.init(
    entity="<project>", 
    project="<project>",
    id="<run-id>"
)
```

You can use a run's unique ID to directly navigate to the run's overview page in the W&B App UI. The proceeding cell shows the URL path for a specific run:

```text title="W&B App URL for a specific run"
https://wandb.ai/<entity>/<project>/<run-id>
```

Where values enclosed in angle brackets (`< >`) are placeholders for the actual values of the entity, project, and run ID.

## Find the unique ID of your run

If you do not specify a run ID when you initialize a run, W&B generates a random run ID. You can find the unique ID of a run in the W&B App UI.

1. Navigate to the W&B App UI at [https://wandb.ai/home](https://wandb.ai/home).
2. Navigate to the W&B project you specified when you initialized the run.
3. Within your project's workspace, you will see a table labeled **Runs**. This table lists all the runs that are in your project. From the list of runs shown, select the run you want to view.
4. Select the **Overview** tab.

Within the **Run path** field, you will find the unique run ID. The unique ID is the last part of the run path. The run path consists of the name of your team, the name of the project, and the run ID. 

For example, in the proceeding image, the unique run ID is `9mxi1arc`:

![](/images/runs/unique-run-id.png)

<!-- ## End a run


W&B automatically ends runs and logs data from that run to your W&B project. You can end a run manually with the [`run.finish`](../../ref/python/run.md#finish) command. For example:

```python
import wandb

run = wandb.init()
run.finish()
```

:::info
W&B suggests that you use the [`wandb.finish`](../../ref/python/finish.md) method at the end of the child process if you call [`wandb.init`](../../ref/python/init.md) from a child process.
::: -->








<!-- ### Search runs

Search for a specific run by name in the sidebar. You can use regex to filter down your visible runs. The search box affects which runs are shown on the graph. Here's an example:

![](/images/app_ui/project_page_search_for_runs.gif)

### Filter runs

### Organize runs -->





