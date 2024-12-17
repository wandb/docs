---
description: Edit a report interactively with the App UI or programmatically with
  the W&B SDK.
menu:
  default:
    identifier: edit-a-report
    parent: reports
title: Edit a report
weight: 2
---

Edit a report interactively with the App UI or programmatically with the W&B SDK.

Reports consist of _blocks_. Blocks make up the body of a report. Within these blocks you can add text, images, embedded visualizations, plots from experiments and runs, and panel grids.

_Panel grids_ are a specific type of block that hold panels and _run sets_. Run sets are a collection of runs logged to a project in W&B. Panels are visualizations of run set data.

{{% alert %}}
Check out the [Programmatic workspaces tutorial](../../tutorials/workspaces.md) for a step-by-step example on how to create and customize a saved workspace view.
{{% /alert %}}

{{% alert %}}
Ensure that you have `wandb-workspaces` installed in addition to the W&B Python SDK if you want to programmatically edit a report:

```bash
pip install wandb wandb-workspaces
```
{{% /alert %}}

### Add plots

Each panel grid has a set of run sets and a set of panels. The run sets at the bottom of the section control what data shows up on the panels in the grid. Create a new panel grid if you want to add charts that pull data from a different set of runs.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. Select **Add panel** to add a panel. You can add any panel that is supported by W&B, including a line plot, scatter plot, or parallel coordinates chart.

{{< img src="/images/reports/demo_report_add_panel_grid.gif" alt="Add charts to a report" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Add plots to a report programmatically with the SDK. Pass a list of one or more plot or chart objects to the `panels` parameter in the `PanelGrid` Public API Class. Create a plot or chart object with its associated Python Class.

The following examples demonstrate how to create a line plot and scatter plot:

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

### Add run sets

Add run sets from projects interactively with the App UI or the W&B SDK.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}

Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown, choose **Panel Grid**. This will automatically import the run set from the project the report was created from.
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Add run sets from projects with the `wr.Runset()` and `wr.PanelGrid` Classes. The following procedure describes how to add a runset:

1. Create a `wr.Runset()` object instance. Provide the name of the project that contains the runsets for the project parameter and the entity that owns the project for the entity parameter.
2. Create a `wr.PanelGrid()` object instance. Pass a list of one or more runset objects to the `runsets` parameter.
3. Store one or more `wr.PanelGrid()` object instances in a list.
4. Update the report instance's `blocks` attribute with the list of panel grid instances.

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
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
    project="report-editing",
    title="An amazing title",
    description="A descriptive description.",
)

panel_grids = wr.PanelGrid(
    panels=[
        wr.LinePlot(x="x", y=["y"]),
        wr.ScatterPlot(x="y", y="y"),
    ],
    runsets=[wr.RunSet(project="<project-name>", entity="<entity-name>")],
)

report.blocks = [panel_grids]
report.save()
```
{{% /tab %}}

{{< /tabpane >}}

### Add code blocks

Add code blocks to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}
Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown choose **Code**.

Select the name of the programming language on the right-hand side of the code block. This will expand a dropdown. From the dropdown, select your programming language syntax. You can choose from JavaScript, Python, CSS, JSON, HTML, Markdown, and YAML.
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Use the `wr.CodeBlock` Class to create a code block programmatically. Provide the name of the language and the code you want to display for the `language` and `code` parameters, respectively.

The following example demonstrates a list in YAML format:

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

The following example demonstrates a Python code block:

```python
report.blocks = [wr.CodeBlock(code=["print('Hello, World!')"], language="python")]
report.save()
```
{{% /tab %}}

{{< /tabpane >}}

### Add markdown

Add markdown to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}
Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown choose **Markdown**.
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Use the `wandb.apis.reports.MarkdownBlock` Class to create a markdown block programmatically. Pass a string to the `text` parameter:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.MarkdownBlock(text="Markdown cell with *italics* and **bold** and $e=mc^2$")
]
report.save()
```
{{% /tab %}}

{{< /tabpane >}}

### Add HTML elements

Add HTML elements to your report interactively with the App UI or with the W&B SDK.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}
Enter a forward slash (`/`) in the report to display a dropdown menu. From the dropdown select a type of text block. For example, to create an H2 heading block, select the `Heading 2` option.
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Pass a list of one or more HTML elements to the `wandb.apis.reports.blocks` attribute. The following example demonstrates how to create an H1, H2, and an unordered list:

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
{{% /tab %}}

{{< /tabpane >}}

### Embed rich media links

Embed rich media within the report with the App UI or with the W&B SDK.

{{< tabpane text=true >}}

{{% tab header="App UI" value="app" %}}

Copy and paste URLs into reports to embed rich media within the report. The following animations demonstrate how to copy and paste URLs from Twitter, YouTube, and SoundCloud.

#### Twitter
Copy and paste a Tweet link URL into a report to view the Tweet within the report.

{{< img src="/images/reports/twitter.gif" alt="" >}}

#### YouTube
Copy and paste a YouTube video URL link to embed a video in the report.

{{< img src="/images/reports/youtube.gif" alt="" >}}

#### SoundCloud
Copy and paste a SoundCloud link to embed an audio file into a report.

{{< img src="/images/reports/soundcloud.gif" alt="" >}}
{{% /tab %}}

{{% tab header="Workspaces API" value="sdk" %}}

Pass a list of one or more embedded media objects to the `wandb.apis.reports.blocks` attribute. The following example demonstrates how to embed video and Twitter media into a report:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

report = wr.Report(project="report-editing")

report.blocks = [
    wr.Video(url="https://www.youtube.com/embed/6riDJMI-Y8U"),
    wr.Twitter(
        embed_html='<blockquote class="twitter-tweet"><p lang="en" dir="ltr">The voice of an angel, truly. <a href="https://twitter.com/hashtag/MassEffect?src=hash&amp;ref_src=twsrc%5Etfw">#MassEffect</a> <a href="https://t.co/nMev97Uw7F">pic.twitter.com/nMev97Uw7F</a></p>&mdash; Mass Effect (@masseffect) <a href="https://twitter.com/masseffect/status/1428748886655569924?ref_src=twsrc%5Etfw">August 20, 2021</a></blockquote>'
    ),
]
report.save()
```
{{% /tab %}}

{{< /tabpane >}}

### Duplicate and delete panel grids

If you have a layout that you would like to reuse, you can select a panel grid and copy-paste it to duplicate it in the same report or even paste it into a different report.

Highlight a whole panel grid section by selecting the drag handle in the upper right corner. Click and drag to highlight and select a region in a report such as panel grids, text, and headings.

{{< img src="/images/reports/demo_copy_and_paste_a_panel_grid_section.gif" alt="" >}}

Select a panel grid and press `delete` on your keyboard to delete a panel grid.

{{< img src="/images/reports/delete_panel_grid.gif" alt="" >}}

### Collapse headers to organize Reports

Collapse headers in a Report to hide content within a text block. When the report is loaded, only headers that are expanded will show content. Collapsing headers in reports can help organize your content and prevent excessive data loading. The following gif demonstrates the process:

{{< img src="/images/reports/collapse_headers.gif" alt="" >}}
