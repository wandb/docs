---
description: Compare runs from two different projects with cross-project reports.
menu:
  default:
    identifier: cross-project-reports
    parent: reports
title: Compare runs across projects
weight: 60
---
{{% alert %}}
Watch a [video demonstrating comparing runs across projects](https://www.youtube.com/watch?v=uD4if_nGrs4) (2 min).
{{% /alert %}}


Compare runs from two different projects with cross-project reports. Use the project selector in the run set table to pick a project.

{{< img src="/images/reports/howto_pick_a_different_project_to_draw_runs_from.gif" alt="Compare runs across different projects" >}}

The visualizations in the section pull columns from the first active runset. Make sure that the first run set checked in the section has that column available if you do not see the metric you are looking for in the line plot.

This feature supports history data on time series lines, but we don't support pulling different summary metrics from different projects. In other words, you can not create a scatter plot from columns that are only logged in another project.

If you need to compare runs from two projects and the columns are not working, add a tag to the runs in one project and then move those runs to the other project. You can still filter only the runs from each project, but the report includes all the columns for both sets of runs.

## View-only report links

Share a view-only link to a report that is in a private project or team project.

{{< img src="/images/reports/magic-links.gif" alt="View-only report links" >}}

View-only report links add a secret access token to the URL, so anyone who opens the link can view the page. Anyone can use the magic link to view the report without logging in first. For customers on [W&B Local]({{< relref "/guides/hosting/" >}}) private cloud installations, these links remain behind your firewall, so only members of your team with access to your private instance _and_ access to the view-only link can view the report.

In **view-only mode**, someone who is not logged in can see the charts and mouse over to see tooltips of values, zoom in and out on charts, and scroll through columns in the table. When in view mode, they cannot create new charts or new table queries to explore the data. View-only visitors to the report link won't be able to click a run to get to the run page. Also, the view-only visitors would not be able to see the share modal but instead would see a tooltip on hover which says: `Sharing not available for view only access`.

{{% alert color="info" %}}
The magic links are only available for “Private” and “Team” projects. For “Public” (anyone can view) or “Open” (anyone can view and contribute runs) projects, the links can't turn on/off because this project is public implying that it is already available to anyone with the link.
{{% /alert %}}

## Send a graph to a report

Send a graph from your workspace to a report to keep track of your progress. Click the dropdown menu on the chart or panel you'd like to copy to a report and click **Add to report** to select the destination report.