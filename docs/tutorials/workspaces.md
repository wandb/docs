import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Programmatic Workspaces

<CTAButtons colabLink='https://colab.research.google.com/github/wandb/wandb-workspaces/blob/Update-wandb-workspaces-tuturial/Workspace_tutorial.ipynb'/>



Organize and visualize your machine learning experiments more effectively by programmatically create, manage, and customize workspaces by defining configurations, setting panel layouts, and organizing sections with the [`wandb-workspaces`](https://github.com/wandb/wandb-workspaces/tree/main) W&B library. Load and modify workspaces with URLs, filter and group runs using expressions, and customize run appearances.

`wandb-workspaces` is a Python library for programmatically creating and customizing W&B [Workspaces](https://docs.wandb.ai/guides/app/pages/workspaces) and [Reports](https://docs.wandb.ai/guides/reports).

In this tutorial you will see how to use `wandb-workspaces` to create and customize workspaces by defining configurations, set panel layouts, and organize sections.


### How to use this notebook
* Run each cell one at a time. 
* Copy and paste the URL that is printed after you run a cell to view the changes made to the workspace.


:::info
Programmatic interaction with workspaces is currently supported for [**Saved workspaces views**](https://docs.wandb.ai/guides/app/pages/workspaces#saved-workspace-views). Saved workspaces views are collaborative snapshots of a workspace. Anyone on your team can view, edit, and save changes to saved workspace views. 
:::

## 1. Install and import dependencies


```python
# Install dependencies
!pip install wandb wandb-workspaces rich
```


```python
# Import dependencies
import os
import wandb
import wandb_workspaces.workspaces as ws
import wandb_workspaces.reports.v2 as wr # We use the Reports API for adding panels

# Improve output formatting
%load_ext rich
```

## 2. Create a new project and workspace

For this tutorial we will create a new project so that we can experiment with the `wandb_workspaces` API:

Note: You can load an existing workspace using its unique `Saved view` URL. See the next code block to see how to do this.


```python
# Initialize Weights & Biases and Login
wandb.login()

# Function to create a new project and log sample data
def create_project_and_log_data():
    project = "workspace-api-example"  # Default project name

    # Initialize a run to log some sample data
    with wandb.init(project=project, name="sample_run") as run:
        for step in range(100):
            wandb.log({
                "Step": step,
                "val_loss": 1.0 / (step + 1),
                "val_accuracy": step / 100.0,
                "train_loss": 1.0 / (step + 2),
                "train_accuracy": step / 110.0,
                "f1_score": step / 100.0,
                "recall": step / 120.0,
            })
    return project

# Create a new project and log data
project = create_project_and_log_data()
entity = wandb.Api().default_entity
```

### (Optional) Load an existing project and workspace
Instead of creating a new project, you can load one of your own existing project and workspace. To do this, find the unique workspace URL and pass it to `ws.Workspace.from_url` as a string. The URL has the form `https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc`. 

For example:

```python
wandb.login()

workspace = ws.Workspace.from_url("https://wandb.ai/[SOURCE-ENTITY]/[SOURCE-USER]?nw=abc").

workspace = ws.Workspace(
    entity="NEW-ENTITY",
    project=NEW-PROJECT,
    name="NEW-SAVED-VIEW-NAME"
)
```

## 3. Programmatic workspace examples
Below are examples for using programmatic workspace features:


```python
# See all available settings for workspaces, sections, and panels.
all_settings_objects = [x for x in dir(ws) if isinstance(getattr(ws, x), type)]
all_settings_objects
```

### Create a workspace with `saved view`
This example demonstrates how to create a new workspace and populate it with sections and panels. Workspaces can be edited like regular Python objects, providing flexibility and ease of use.


```python
def sample_workspace_saved_example(entity: str, project: str) -> str:
    workspace: ws.Workspace = ws.Workspace(
        name="Example W&B Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Validation Metrics",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.BarPlot(metrics=["val_accuracy"]),
                    wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                ],
                is_open=True,
            ),
        ],
    )
    workspace.save()
    print("Sample Workspace saved.")
    return workspace.url

workspace_url: str = sample_workspace_saved_example(entity, project)
```

### Load a workspace from a URL
Duplicate and customize workspaces without affecting the original setup. To do this, load an existing workspace and save it as a new view:


```python
def save_new_workspace_view_example(url: str) -> None:
    workspace: ws.Workspace = ws.Workspace.from_url(url)

    workspace.name = "Updated Workspace Name"
    workspace.save()

    print(f"Workspace saved as new view.")

save_new_workspace_view_example(workspace_url)
```

Note that your workspace is now named "Updated Workspace Name".

### Basic settings
The following code shows how to create a workspace, add sections with panels, and configure settings for the workspace, individual sections, and panels:


```python
# Function to create and configure a workspace with custom settings
def custom_settings_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(name="An example workspace", entity=entity, project=project)
    workspace.sections = [
        ws.Section(
            name="Validation",
            panels=[
                wr.LinePlot(x="Step", y=["val_loss"]),
                wr.LinePlot(x="Step", y=["val_accuracy"]),
                wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                wr.ScalarChart(metric="recall", groupby_aggfunc="mean"),
            ],
            is_open=True,
        ),
        ws.Section(
            name="Training",
            panels=[
                wr.LinePlot(x="Step", y=["train_loss"]),
                wr.LinePlot(x="Step", y=["train_accuracy"]),
            ],
            is_open=False,
        ),
    ]

    workspace.settings = ws.WorkspaceSettings(
        x_axis="Step",
        x_min=0,
        x_max=75,
        smoothing_type="gaussian",
        smoothing_weight=20.0,
        ignore_outliers=False,
        remove_legends_from_panels=False,
        tooltip_number_of_runs="default",
        tooltip_color_run_names=True,
        max_runs=20,
        point_visualization_method="bucketing",
        auto_expand_panel_search_results=False,
    )

    section = workspace.sections[0]
    section.panel_settings = ws.SectionPanelSettings(
        x_min=25,
        x_max=50,
        smoothing_type="none",
    )

    panel = section.panels[0]
    panel.title = "Validation Loss Custom Title"
    panel.title_x = "Custom x-axis title"

    workspace.save()
    print("Workspace with custom settings saved.")

# Run the function to create and configure the workspace
custom_settings_example(entity, project)
```

Note that you are now viewing a different saved view called "An example workspace".

## Customize runs
The following code cells show you how to filter, change the color, group, and sort runs programmatically. 

In each example, the general workflow is to specify the desired customization as an argument to the appropiate parameter in `ws.RunsetSettings`.

### Filter runs
You can create filters with python expressions and metrics you log with `wandb.log` or that are logged automatically as part of the run such as **Created Timestamp**.  You can also reference filters by how they appear in the W&B App UI such as the **Name**, **Tags**, or **ID**.

The following example shows how to filter runs based on the validation loss summary, validation accuracy summary, and the regex specified:


```python
def advanced_filter_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Apply multiple filters: val_loss < 0.1, val_accuracy > 0.8, and run name matches regex pattern
    workspace: ws.Workspace = ws.Workspace(
        name="Advanced Filtered Workspace with Regex",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Advanced Filtered Section",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            filters=[
                (ws.Summary("val_loss") < 0.1),  # Filter runs by the 'val_loss' summary
                (ws.Summary("val_accuracy") > 0.8),  # Filter runs by the 'val_accuracy' summary
                (ws.Metric("ID").isin([run.id for run in wandb.Api().runs(f"{entity}/{project}")])),
            ],
            regex_query=True,
        )
    )

    # Add regex search to match run names starting with 's'
    workspace.runset_settings.query = "^s"
    workspace.runset_settings.regex_query = True

    workspace.save()
    print("Workspace with advanced filters and regex search saved.")

advanced_filter_example(entity, project)
```

Note that passing in a list of filter expressions applies the boolean "AND" logic.

### Change the colors of runs
This example demonstrates how to change the colors of the runs in a workspace:


```python
def run_color_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Dynamically assign colors to the runs
    run_colors: list = ['purple', 'orange', 'teal', 'magenta']
    run_settings: dict = {}
    for i, run in enumerate(runs):
        run_settings[run.id] = ws.RunSettings(color=run_colors[i % len(run_colors)])

    workspace: ws.Workspace = ws.Workspace(
        name="Run Colors Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Run Colors Section",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            run_settings=run_settings
        )
    )

    workspace.save()
    print("Workspace with run colors saved.")

run_color_example(entity, project)
```

### Group runs

This example demonstrates how to group runs by specific metrics.



```python
def grouping_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(
        name="Grouped Runs Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Grouped Runs",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            groupby=[ws.Metric("Name")]
        )
    )
    workspace.save()
    print("Workspace with grouped runs saved.")

grouping_example(entity, project)
```

### Sort runs
This example demonstrates how to sort runs based on the validation loss summary:


```python
def sorting_example(entity: str, project: str) -> None:
    workspace: ws.Workspace = ws.Workspace(
        name="Sorted Runs Workspace",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Sorted Runs",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                ],
                is_open=True,
            ),
        ],
        runset_settings=ws.RunsetSettings(
            order=[ws.Ordering(ws.Summary("val_loss"))] #Order using val_loss summary
        )
    )
    workspace.save()
    print("Workspace with sorted runs saved.")

sorting_example(entity, project)
```

## 4. Putting it all together: comprehenive example

This example demonstrates how to create a comprehensive workspace, configure its settings, and add panels to sections:


```python
def full_end_to_end_example(entity: str, project: str) -> None:
    # Get all runs in the project
    runs: list = wandb.Api().runs(f"{entity}/{project}")

    # Dynamically assign colors to the runs and create run settings
    run_colors: list = ['red', 'blue', 'green', 'orange', 'purple', 'teal', 'magenta', '#FAC13C']
    run_settings: dict = {}
    for i, run in enumerate(runs):
        run_settings[run.id] = ws.RunSettings(color=run_colors[i % len(run_colors)], disabled=False)

    workspace: ws.Workspace = ws.Workspace(
        name="My Workspace Template",
        entity=entity,
        project=project,
        sections=[
            ws.Section(
                name="Main Metrics",
                panels=[
                    wr.LinePlot(x="Step", y=["val_loss"]),
                    wr.LinePlot(x="Step", y=["val_accuracy"]),
                    wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
                ],
                is_open=True,
            ),
            ws.Section(
                name="Additional Metrics",
                panels=[
                    wr.ScalarChart(metric="precision", groupby_aggfunc="mean"),
                    wr.ScalarChart(metric="recall", groupby_aggfunc="mean"),
                ],
            ),
        ],
        settings=ws.WorkspaceSettings(
            x_axis="Step",
            x_min=0,
            x_max=100,
            smoothing_type="none",
            smoothing_weight=0,
            ignore_outliers=False,
            remove_legends_from_panels=False,
            tooltip_number_of_runs="default",
            tooltip_color_run_names=True,
            max_runs=20,
            point_visualization_method="bucketing",
            auto_expand_panel_search_results=False,
        ),
        runset_settings=ws.RunsetSettings(
            query="",
            regex_query=False,
            filters=[
                ws.Summary("val_loss") < 1,
                ws.Metric("Name") == "sample_run",
            ],
            groupby=[ws.Metric("Name")],
            order=[ws.Ordering(ws.Summary("Step"), ascending=True)],
            run_settings=run_settings
        )
    )
    workspace.save()
    print("Workspace created and saved.")

full_end_to_end_example(entity, project)
```
