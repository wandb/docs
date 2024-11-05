---
displayed_sidebar: default
title: Initialize a run
---


Initialize a W&B run with [`wandb.init()`](../../ref/python/init.md). The proceeding code snippet shows how to import the W&B Python SDK and initialize a run:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
```

When you initialize a run, W&B logs your run to the project you specify for the project field (`wandb.init(project="<project>"`). W&B creates a new project if the project does not already exist. If the project already exists, W&B stores the run in that project.


:::info
If you do not specify a project name, W&B stores the run in a project called "Uncategorized".
:::


Each run in W&B has a [unique identifier known as a *run ID*](#specify-a-unique-run-id). [You can specify a unique ID](#specify-a-unique-run-id) or let [W&B randomly generate one for you](#find-generated-run-id).

Each run also has a human-readable,[ non-unique identifier known as a *run name*](#name-your-run). You can specify a name for your run or let W&B randomly generate one for you.


For example, consider the proceeding code snippet: 

```python
import wandb

run = wandb.init(entity="wandbee", project="awesome-project")
run.finish()
```

The code snippet above produces this output:
![](/images/runs/run_example.png)

Where `wandbee` is the entity that logged the run, `awesome-project` is the name of the project the run is logged to, `likely-lion-9` is the name of the run, and `xlm66ixq` is the run ID.


## Specify a unique run ID

Run IDs are unique identifiers for runs. By default, W&B generates a random and unique run ID for you when you initialize a new run. 

You can specify your own run ID by passing the `id` parameter to the [`wandb.init`](../../ref/python/init.md) method. 

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

## Find generated run ID

If you do not specify a run ID when you initialize a run, W&B generates a random run ID for you. You can find the unique ID of a run in the W&B App UI.

1. Navigate to the W&B App UI at [https://wandb.ai/home](https://wandb.ai/home).
2. Navigate to the W&B project you specified when you initialized the run.
3. Within your project's workspace, you will see a table labeled **Runs**. This table lists all the runs that are in your project. From the list of runs shown, select the run you want to view.
4. Select the **Overview** tab.

Within the **Run path** field, you will find the unique run ID. The unique ID is the last part of the run path. The run path consists of the name of your team, the name of the project, and the run ID. 

For example, in the proceeding image, the unique run ID is `9mxi1arc`:

![](/images/runs/unique-run-id.png)


## Name your run 

The name of a run is a human-readable, non-unique identifier. 

By default, W&B generates a random run name when you initialize a new run. The name of a run appears within your project's workspace and at the top of the [run's overview page](./view-runs.md#overview-tab).

:::tip
Use run names as a way to quickly identify a run in your project workspace.
:::

You can specify a name for your run by passing the `name` parameter to the [`wandb.init`](../../ref/python/init.md) method. 


```python 
import wandb

run = wandb.init(
    entity="<project>", 
    project="<project>", 
    name="<run-name>"
)
```







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





