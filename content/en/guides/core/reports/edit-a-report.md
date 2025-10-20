---
description: Edit a report interactively with the App UI or programmatically with
  the W&B SDK.
menu:
  default:
    identifier: edit-a-report
    parent: reports
title: Edit a report
weight: 20
---

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

Reports consist of _blocks_. Blocks make up the body of a report. Within these blocks you can add text, images, embedded visualizations, plots from experiments and run, and panels grids.

_Panel grids_ are a specific type of block that hold panels and _run sets_. Run sets are a collection of runs logged to a project in W&B. Panels are visualizations of run set data.


Edit a report interactively with the W&B App or programmatically with the W&B Python SDK.

<!-- {{% alert %}}
Check out the [Programmatic workspaces tutorial]({{< relref "/tutorials/workspaces.md" >}}) for a step by step example on how create and customize a saved workspace view.
{{% /alert %}} -->

{{% alert title="Programmatic editing requirements" %}}

To programmatically edit a report, you need to install the W&B Report and Workspace API `wandb-workspaces` in addition to the W&B Python SDK (`wandb`):

{{< code language="shell" source="/bluehawk/snippets/wandb_install.snippet.pip_install_wandb_packages.sh" >}}

Within your Python script or notebook, import both the W&B Python SDK (`wandb`) and the `wandb_workspaces.reports.v2` module to access the Report and Workspace API:

{{< code language="python" source="/bluehawk/snippets/import_wandb.snippet.import_wandb_and_workspaces.py" >}}

Throughtout this guide, code snippets that demonstrate how to programmatically edit a report are prefixed with `wr.` to indicate they are part of the Report and Workspace API.

{{% /alert %}}

## Add plots

Each panel grid has a set of run sets and a set of panels. The run sets at the bottom of the section control what data shows up on the panels in the grid. Create a new panel grid if you want to add charts that pull data from a different set of runs.

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. Select **Add panel** to add a panel. You can add any panel that is supported by W&B, including a line plot, scatter plot or parallel coordinates chart.

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="Add charts to a report" >}}
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}
Add plots to a report programmatically with the SDK. Pass a list of one or more plot or chart objects to the `panels` parameter in the `PanelGrid` Public API Class. Create a plot or chart object with its associated Python Class.


The following examples demonstrates how to create a line plot and scatter plot.

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-plots.py" >}}

{{% /tab %}}
{{< /tabpane >}}


## Add run sets

Add run sets from projects interactively with the App UI or the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown, choose **Panel Grid**. This will automatically import the run set from the project the report was created from.

If you import a panel into a report, run names are inherited from the project. In the report, you can optionally [rename a run]({{< relref "/guides/models/track/runs/#rename-a-run" >}}) to give the reader more context. The run is renamed only in the individual panel. If you clone the panel in the same report, the run is also renamed in the cloned panel.

1. In the report, click the pencil icon to open the report editor.
1. In the run set, find the run to rename. Hover over the report name, click the three vertical dots. Select one of the following choices, then submit the form.

    - **Rename run for project**: rename the run across the entire project. To generate a new random name, leave the field blank.
    - **Rename run for panel grid** rename the run only in the report, preserving the existing name in other contexts. Generating a new random name is not supported.

1. Click **Publish report**.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

Add run sets from projects with the `wr.Runset()` and `wr.PanelGrid` Classes. The following procedure describes how to add a runset:

1. Create a `wr.Runset()` object instance. Provide the name of the project that contains the run sets for the project parameter and the entity that owns the project for the entity parameter.
2. Create a `wr.PanelGrid()` object instance. Pass a list of one or more runset objects to the `run sets` parameter.
3. Store one or more `wr.PanelGrid()` object instances in a list.
4. Update the report instance blocks attribute with the list of panel grid instances.


{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-runset-no-panels.py" >}}

{{% /tab %}}
{{< /tabpane >}}

## Add run sets and panels
You can optionally add runsets and panels with one call to the SDK:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-runsets-and-panels.py" >}}

## Freeze a run set

A report automatically updates run sets to show the latest data from the project. You can preserve the run set in a report by *freezing* that run set. When you freeze a run set, you preserve the state of the run set in a report at a point in time.

To freeze a run set when viewing a report, click the snowflake icon in its panel grid near the **Filter** button.

{{< img src="/images/reports/freeze_runset.png" alt="Freeze runset button" >}}

## Group a run set programmatically

Group runs in a run set programmatically with the [Workspace and Reports API]({{< relref "/ref/wandb_workspaces/reports" >}}).

You can group runs in a run set by config values, run metadata or summary metrics. The following table lists the available grouping methods along with the available keys for that grouping method:

| Grouping Method | Description |Available keys |
| ---|------| --- |
| Config values| Group runs by config values | Values specified in config parameter in `wandb.init(config=)` |
| Run metadata| Group runs by run metadata | `State`, `Name`, `JobType` |
| Summary metrics| Group runs by summary metrics | Values you log to a run with `wandb.Run.log()` |

<!-- Key differences between grouping runs in a report and grouping runs in a workspace:
1. String paths vs objects: Reports use string paths like `"config.group"` instead of type objects like `ws.Config("group")`.
2. Dot notation: Use dots to access nested values: `"config.model_type"`.
3. Predefined fields: Some fields like `"Name"`, `"Tags"`, `"State"` are special run metadata fields. -->

### Group runs by config values

Group runs by config values to compare runs with similar configurations. Config values are parameters you specify in your run configuration `(wandb.init(config=))`. To group runs by config values, use the `config.<key>` syntax, where `<key>` is the name of the config value you want to group by. 

For example, the following code snippet first initializes a run with a config value for `group`, then groups runs in a report based on the `group` config value. Replace values for `<entity>` and `<project>` with your W&B entity and project names.


{{< code language="python" source="/bluehawk/snippets/group_runs.snippet.group_runs.py" >}}



You can then group runs by the `config.group` value:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.group-runs-config.py" >}}


### Group runs by run metadata

Group runs by a run's name (`Name`), state (`State`), or job type (`JobType`). 

Continuing from the previous example, you can group your runs by their name with the following code snippet:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.group-runs-metadata.py" >}}

{{% alert %}}
The name of the run is the name you specify in the `wandb.init(name=)` parameter. If you do not specify a name, W&B generates a random name for the run.

You can find the name of the run in the **Overview** page of a run in the W&B App or programmatically with `Api.runs().run.name`.
{{% /alert %}}

### Group runs by summary metrics

The following examples demonstrate how to group runs by summary metrics. Summary metrics are the values you log to a run with `wandb.Run.log()`. After you log a run, you can find the names of your summary metrics in the W&B App under the **Summary** section of a run's **Overview** page.

The syntax for grouping runs by summary metrics is `summary.<key>`, where `<key>` is the name of the summary metric you want to group by. 

For example, suppose you log a summary metric called `acc`:

{{< code language="python" source="/bluehawk/snippets/group_runs.snippet.group_runs.py" >}}

You can then group runs by the `summary.acc` summary metric:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.group-runs-summary-metrics.py" >}}

## Filter a run set programmatically

Programmatically filter run sets and add them to a report with the [Workspace and Reports API]({{< relref "/ref/wandb_workspaces/reports" >}}).

The general syntax for a filter expression is:

```text
Filter('key') operation <value>
```

Where `key` is the name of the filter, `operation` is a comparison operator (e.g., `>`, `<`, `==`, `in`, `not in`, `or`, and `and`), and `<value>` is the value to compare against. `Filter` is a placeholder for the type of filter you want to apply. The following table lists the available filters and their descriptions:

| Filter | Description | Available keys |
| ---|---| --- |
|`Config('key')` | Filter by config values | Values specified in `config` parameter in `wandb.init(config=)`. |
|`SummaryMetric('key')` | Filter by summary metrics | Values you log to a run with `wandb.Run.log()`. |
|`Tags('key')` | Filter by tags | Tag values that you add to your run (programmatically or with the W&B App). |
|`Metric('key')` | Filter by run properties | `tags`, `state`, `displayName`, `jobType` |

Once you have defined your filters, you can create a report and pass the filtered run sets to `wr.PanelGrid(runsets=)`. See the **Report and Workspace API** tabs throughout this page for more information on how to add various elements to a report programmatically.

The following examples demonstrate how to filter run sets in a report. Replace values enclosed in `<>` with your own values.

### Config filters

Filter a runset by one or more config values. Config values are parameters you specify in your run configuration (`wandb.init(config=)`).

For example, the following code snippet first initializes a run with a config value for `learning_rate` and `batch_size`, then filters runs in a report based on the `learning_rate` config value.

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.config-filters-0.py" >}}

The following code snippet shows how to filter runs based on learning rates greater than `0.01`:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.config-filters-1.py" >}}

The following code snippet shows how to filter runs based on a single config value that have a learning rate greater than `0.01`and a batch size equal to `32`:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.config-filters-2.py" >}} 

Once you have defined your filtered run set, you can create a report and pass the filtered run set to `wr.PanelGrid(runsets=)`:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.config-filters-3.py" >}}

### Metric filters

Filter run sets based on a run's: tag (`tags`), run state (`state`), run name (`displayName`), or job type (`jobType`).

{{% alert %}}
`Metric` filters posses a different syntax. Pass a list of values as a list.

```text
Metric('key') operation [<value>]
```
{{% /alert %}}

Consider the following Python snippet that creates three runs and assigns each of them a name:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.metric-filters-0.py" >}}

When you create your report, you can filter runs by their display name. For example, to filter runs with names `run1`, `run2`, and `run3`, you can use the following code:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.metric-filters-1.py" >}}


{{% alert %}}
You can find the name of the run in the **Overview** page of a run in the W&B App or programmatically with `Api.runs().run.name`.
{{% /alert %}}

The following examples demonstrate how to filter a runset by the run's state (`finished`, `crashed`, or `running`). In the following example, we filter a run set to include only runs that have finished:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.metric-filters-2.py" >}}

The following example demonstrates how to filter a run set to exclude runs that have crashed:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.metric-filters-3.py" >}}


### SummaryMetric filters

The following examples demonstrate how to filter a run set by summary metrics. Summary metrics are the values you log to a run with `wandb.Run.log()`. After you log a run, you can find the names of your summary metrics in the W&B App under the **Summary** section of a run's **Overview** page.

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.summary-metric-filters.py" >}}

### Tags filters

The following code snippet shows how to filter a runs set by its tags. Tags are values you add to a run (programmatically or with the W&B App).

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.tag-filters.py" >}}

## Add code blocks

Add code blocks to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown choose **Code**.

Select the name of the programming language on the right hand of the code block. This will expand a dropdown. From the dropdown, select your programming language syntax. You can choose from Javascript, Python, CSS, JSON, HTML, Markdown, and YAML.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

Use the `wr.CodeBlock` Class to create a code block programmatically. Provide the name of the language and the code you want to display for the language and code parameters, respectively.

For example the following example demonstrates a list in YAML file:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-code-blocks.py" >}}


This will render a code block similar to:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

The following example demonstrates a Python code block:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-code-block-python.py" >}}

This will render a code block similar to:

```md
Hello, World!
```

{{% /tab %}}

{{% /tabpane %}}

## Add markdown

Add markdown to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown choose **Markdown**.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

Use the `wandb.apis.reports.MarkdownBlock` Class to create a markdown block programmatically. Pass a string to the `text` parameter:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-markdown.py" >}}

<br>

This will render a markdown block similar to:

{{< img src="/images/reports/markdown.png" alt="Rendered markdown block" >}}

{{% /tab %}}

{{% /tabpane %}}


## Add HTML elements

Add HTML elements to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown select a type of text block. For example, to create an H2 heading block, select the `Heading 2` option.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

Pass a list of one or more HTML elements to `wandb.apis.reports.blocks` attribute. The following example demonstrates how to create an H1, H2, and an unordered list:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.add-html.py" >}}


This will render a HTML elements  to the following:


{{< img src="/images/reports/render_html.png" alt="Rendered HTML elements" >}}

{{% /tab %}}

{{% /tabpane %}}

## Embed rich media links

Embed rich media within the report with the App UI or with the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Copy and past URLs into reports to embed rich media within the report. The following animations demonstrate how to copy and paste URLs from Twitter, YouTube, and SoundCloud.

### Twitter

Copy and paste a Tweet link URL into a report to view the Tweet within the report.

{{< img src="/images/reports/twitter.gif" alt="Embedding Twitter content" >}}

### Youtube

Copy and paste a YouTube video URL link to embed a video in the report.

{{< img src="/images/reports/youtube.gif" alt="Embedding YouTube videos" >}}

### SoundCloud

Copy and paste a SoundCloud link to embed an audio file into a report.

{{< img src="/images/reports/soundcloud.gif" alt="Embedding SoundCloud audio" >}}

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

Pass a list of one or more embedded media objects to the `wandb.apis.reports.blocks` attribute. The following example demonstrates how to embed video and Twitter media into a report:

{{< code language="python" source="/bluehawk/snippets/edit-a-report.snippet.embed-rich-media.py" >}}

{{% /tab %}}

{{% /tabpane %}}

## Duplicate and delete panel grids

If you have a layout that you would like to reuse, you can select a panel grid and copy-paste it to duplicate it in the same report or even paste it into a different report.

Highlight a whole panel grid section by selecting the drag handle in the upper right corner. Click and drag to highlight and select a region in a report such as panel grids, text, and headings.

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="Copying panel grids" >}}

Select a panel grid and press `delete` on your keyboard to delete a panel grid.

{{< img src="/images/reports/delete_panel_grid.gif" alt="Deleting panel grids" >}}

## Collapse headers to organize Reports

Collapse headers in a Report to hide content within a text block. When the report is loaded, only headers that are expanded will show content. Collapsing headers in reports can help organize your content and prevent excessive data loading. The following gif demonstrates the process.

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## Visualize relationships across multiple dimensions

To effectively visualize relationships across multiple dimensions, use a color gradient to represent one of the variables. This enhances clarity and makes patterns easier to interpret.

1. Choose a variable to represent with a color gradient (e.g., penalty scores, learning rates, etc.). This allows for a clearer understanding of how penalty (color) interacts with reward/side effects (y-axis) over training time (x-axis).
2. Highlight key trends. Hovering over a specific group of runs highlights them in the visualization.

