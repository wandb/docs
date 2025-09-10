---
title: Reports and Workspaces API
no_list: true
weight: 9
---

The W&B Reports and Workspaces API, accessible at `wandb_workspaces`, allows you to create [reports]({{< relref "/guides/core/reports" >}}), which can be published on the web to share findings, and well as customize a [workspace]({{< relref "/guides/models/app/features/cascade-settings" >}}) where training and fine-tuning work was done. 


{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/" >}}

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

## Installation and setup

### Sign up and create an API key

To authenticate your machine with W&B, you must first generate an API key at https://wandb.ai/authorize.

### Install and import packages

Install the W&B Report and Workspaces library.

```python
pip install wandb-workspaces
```

### Create a report

To create a report, specify your team's entity and provide a name for your report. Replace enclosed text with your values:

```python
import wandb_workspaces.reports.v2 as wr 
# Create
report = wr.Report(
    entity="<team_entity>",
    project="<project_name>",
    title='Quickstart Report',
    description="That was easy!"
)

# Save report
report.save()
```

Next, add blocks and panels to your report. For example, the following code creates a report with a table of contents, header, and a paragraph:

```python
report.blocks = [
    wr.TableOfContents(),
    wr.H1("Text and images example"),
    wr.P("Lorem ipsum dolor sit amet."),
]
report.save()
```

See the [Reports API Quickstart](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Report_API_Quickstart.ipynb) Google Colab for an end to end example.

### Create a workspace

The following code shows how to create a workspace with a section containing three panels: a line plot, a bar plot, and a scalar chart. Replace enclosed text with your values:

```python
# How to import
import wandb_workspaces.workspaces as ws

# Create a workspace
ws.Workspace(
     entity="<team_entity>", # entity that owns the workspace
     project="<project_name>", # project that the workspace is associated with
     sections=[
         ws.Section(
             name="<Validation Metrics>",
             panels=[
                 wr.LinePlot(x="Step", y=["<val_loss>"]),
                 wr.BarPlot(metrics=["<val_accuracy>"]),
                 wr.ScalarChart(metric="<f1_score>", groupby_aggfunc="<mean>"),
             ],
             is_open=True,
         ),
     ],
)
workspace.save()
```

See the [Workspace API Quickstart](https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb#scrollTo=MmxL0wjvrNtQ) Google Colab for an end to end example.

## Classes

[`reports`](./reports.md)

[`workspaces`](./workspaces.md)

