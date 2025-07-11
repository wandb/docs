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

Edit a report interactively with the App UI or programmatically with the W&B SDK.

Reports consist of _blocks_. Blocks make up the body of a report. Within these blocks you can add text, images, embedded visualizations, plots from experiments and run, and panels grids.

_Panel grids_ are a specific type of block that hold panels and _run sets_. Run sets are a collection of runs logged to a project in W&B. Panels are visualizations of run set data.


{{% alert %}}
Check out the [Programmatic workspaces tutorial]({{< relref "/tutorials/workspaces.md" >}}) for a step by step example on how create and customize a saved workspace view.
{{% /alert %}}

{{% alert %}}
Verify that you have the W&B Report and Workspace API `wandb-workspaces` installed in addition to the W&B Python SDK if you want to programmatically edit a report:

```pip
pip install wandb wandb-workspaces
```
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


The proceeding examples demonstrates how to create a line plot and scatter plot.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

blocks = [
    wr.PanelGrid(
        panels=[
            wr.LinePlot(x="time", y="velocity"),
            wr.ScatterPlot(x="time", y="acceleration"),
        ]
    )
]

report.blocks = blocks
report.save()
```

For more information about available plots and charts you can add to a report programmatically, see `wr.panels`.

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

Add run sets from projects with the `wr.Runset()` and `wr.PanelGrid` Classes. The proceeding procedure describes how to add a runset:

1. Create a `wr.Runset()` object instance. Provide the name of the project that contains the runsets for the project parameter and the entity that owns the project for the entity parameter.
2. Create a `wr.PanelGrid()` object instance. Pass a list of one or more runset objects to the `runsets` parameter.
3. Store one or more `wr.PanelGrid()` object instances in a list.
4. Update the report instance blocks attribute with the list of panel grid instances.

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")]
)

report.blocks = [panel_grids]
report.save()
```

You can optionally add runsets and panels with one call to the SDK:

```python
import wandb

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    panels=[
        wr.LinePlot(
            title="line title",
            x="x",
            y=["y"],
            range_x=[0, 100],
            range_y=[0, 100],
            log_x=True,
            log_y=True,
            title_x="x axis title",
            title_y="y axis title",
            ignore_outliers=True,
            groupby="hyperparam1",
            groupby_aggfunc="mean",
            groupby_rangefunc="minmax",
            smoothing_factor=0.5,
            smoothing_type="gaussian",
            smoothing_show_original=True,
            max_runs_to_show=10,
            plot_type="stacked-area",
            font_size="large",
            legend_position="west",
        ),
        wr.ScatterPlot(
            title="scatter title",
            x="y",
            y="y",
            # z='x',
            range_x=[0, 0.0005],
            range_y=[0, 0.0005],
            # range_z=[0,1],
            log_x=False,
            log_y=False,
            # log_z=True,
            running_ymin=True,
            running_ymean=True,
            running_ymax=True,
            font_size="small",
            regression=True,
        ),
    ],
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")],
)


report.blocks = [panel_grids]
report.save()
``` 

{{% /tab %}}
{{< /tabpane >}}


## Freeze a run set

A report automatically updates run sets to show the latest data from the project. You can preserve the run set in a report by *freezing* that run set. When you freeze a run set, you preserve the state of the run set in a report at a point in time.

To freeze a run set when viewing a report, click the snowflake icon in its panel grid near the **Filter** button.

{{< img src="/images/reports/freeze_runset.png" alt="Freeze runset button" >}}

## Add code blocks

Add code blocks to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}
{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown choose **Code**.

Select the name of the programming language on the right hand of the code block. This will expand a dropdown. From the dropdown, select your programming language syntax. You can choose from Javascript, Python, CSS, JSON, HTML, Markdown, and YAML.

{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api" %}}

Use the `wr.CodeBlock` Class to create a code block programmatically. Provide the name of the language and the code you want to display for the language and code parameters, respectively.

For example the proceeding example demonstrates a list in YAML file:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.CodeBlock(
        code=["this:", "- is", "- a", "cool:", "- yaml", "- file"], language="yaml"
    )
]

report.save()
```

This will render a code block similar to:

```yaml
this:
- is
- a
cool:
- yaml
- file
```

The proceeding example demonstrates a Python code block:

```python
report = wr.Report(project="report-editing")


report.blocks = [wr.CodeBlock(code=["Hello, World!"], language="python")]

report.save()
```

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

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
```

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

Pass a list of one or more HTML elements to `wandb.apis.reports.blocks` attribute. The proceeding example demonstrates how to create an H1, H2, and an unordered list:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.H1(text="How Programmatic Reports work"),
    wr.H2(text="Heading 2"),
    wr.UnorderedList(items=["Bullet 1", "Bullet 2"]),
]

report.save()
```

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

Pass a list of one or more embedded media objects to the `wandb.apis.reports.blocks` attribute. The proceeding example demonstrates how to embed video and Twitter media into a report:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>\n'
    ),
]
report.save()
```

{{% /tab %}}

{{% /tabpane %}}

## Duplicate and delete panel grids

If you have a layout that you would like to reuse, you can select a panel grid and copy-paste it to duplicate it in the same report or even paste it into a different report.

Highlight a whole panel grid section by selecting the drag handle in the upper right corner. Click and drag to highlight and select a region in a report such as panel grids, text, and headings.

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="Copying panel grids" >}}

Select a panel grid and press `delete` on your keyboard to delete a panel grid.

{{< img src="/images/reports/delete_panel_grid.gif" alt="Deleting panel grids" >}}

## Collapse headers to organize Reports

Collapse headers in a Report to hide content within a text block. When the report is loaded, only headers that are expanded will show content. Collapsing headers in reports can help organize your content and prevent excessive data loading. The proceeding gif demonstrates the process.

{{< img src="/images/reports/collapse_headers.gif" alt="Collapsing headers in a report." >}}

## Visualize relationships across multiple dimensions

To effectively visualize relationships across multiple dimensions, use a color gradient to represent one of the variables. This enhances clarity and makes patterns easier to interpret.

1. Choose a variable to represent with a color gradient (e.g., penalty scores, learning rates, etc.). This allows for a clearer understanding of how penalty (color) interacts with reward/side effects (y-axis) over training time (x-axis).
2. Highlight key trends. Hovering over a specific group of runs highlights them in the visualization.
