---
description: Export a W&B Report as a PDF or LaTeX.
menu:
  default:
    identifier: clone-and-export-reports
    parent: reports
title: Clone and export reports
weight: 40
---

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

## Export reports

Export a report as a PDF or LaTeX. Within your report, select the kebab icon to expand the dropdown menu. Choose **Download and** select either PDF or LaTeX output format.

## Cloning reports

{{< tabpane text=true >}}
{{% tab header="W&B App" value="app" %}}
Within your report, select the kebab icon to expand the dropdown menu. Choose the **Clone this report** button. Pick a destination for your cloned report in the modal. Choose **Clone report**.

{{< img src="/images/reports/clone_reports.gif" alt="Cloning reports" >}}

Clone a report to reuse a project's template and format. Cloned projects are visible to your team if you clone a project within the team's account. Projects cloned within an individual's account are only visible to that user.
{{% /tab %}}

{{% tab header="Report and Workspace API" value="python_wr_api"%}}

Load a Report from a URL to use it as a template.

```python
report = wr.Report(
    project=PROJECT, title="Quickstart Report", description="That was easy!"
)  # Create
report.save()  # Save
new_report = wr.Report.from_url(report.url)  # Load
```

Edit the content within `new_report.blocks`.

```python
pg = wr.PanelGrid(
    runsets=[
        wr.Runset(ENTITY, PROJECT, "First Run Set"),
        wr.Runset(ENTITY, PROJECT, "Elephants Only!", query="elephant"),
    ],
    panels=[
        wr.LinePlot(x="Step", y=["val_acc"], smoothing_factor=0.8),
        wr.BarPlot(metrics=["acc"]),
        wr.MediaBrowser(media_keys="img", num_columns=1),
        wr.RunComparer(diff_only="split", layout={"w": 24, "h": 9}),
    ],
)
new_report.blocks = (
    report.blocks[:1] + [wr.H1("Panel Grid Example"), pg] + report.blocks[1:]
)
new_report.save()
```
{{% /tab %}}
{{< /tabpane >}}
