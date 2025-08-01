---
description: Compare versions of your model, explore results in a scratch workspace,
  and export findings to a report to save notes and visualizations
menu:
  default:
    identifier: project-page
    parent: experiments
title: Projects
weight: 3
---

A *project* is a central location where you visualize results, compare experiments, view and download artifacts, create an automation, and more. 

{{% alert %}}
Each project has a visibility setting that determines who can access it. For more information about who can access a project, see [Project visibility]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}).
{{% /alert %}}

Each project contains the following tabs:

* [Overview]({{< relref "project-page.md#overview-tab" >}}): snapshot of your project
* [Workspace]({{< relref "project-page.md#workspace-tab" >}}): personal visualization sandbox
* [Runs]({{< relref "#runs-tab" >}}): A table that lists all the runs in your project
* [Automations]({{< relref "#automations-tab">}}): Automations configured in your project
* [Sweeps]({{< relref "project-page.md#sweeps-tab" >}}): automated exploration and optimization
* [Reports]({{< relref "project-page.md#reports-tab" >}}): saved snapshots of notes, runs, and graphs
* [Artifacts]({{< relref "#artifacts-tab" >}}): Contains all runs and the artifacts associated with that run

## Overview tab

* **Project name**: The name of the project. W&B creates a project for you when you initialize a run with the name you provide for the project field. You can change the name of the project at any time by selecting the **Edit** button in the upper right corner.
* **Description**: A description of the project.
* **Project visibility**: The visibility of the project. The visibility setting that determines who can access it. See [Project visibility]({{< relref "/guides/hosting/iam/access-management/restricted-projects.md" >}}) for more information.
* **Last active**: Timestamp of the last time data is logged to this project
* **Owner**: The entity that owns this project
* **Contributors**: The number of users that contribute to this project
* **Total runs**: The total number of runs in this project
* **Total compute**: we add up all the run times in your project to get this total
* **Undelete runs**: Click the dropdown menu and click "Undelete all runs" to recover deleted runs in your project.
* **Delete project**: click the dot menu in the right corner to delete a project

[View a live example](https://app.wandb.ai/example-team/sweep-demo/overview)

{{< img src="/images/track/overview_tab_image.png" alt="Project overview tab" >}}


## Workspace tab

A project's *workspace* gives you a personal sandbox to compare experiments. Use projects to organize models that can be compared, working on the same problem with different architectures, hyperparameters, datasets, preprocessing etc.


**Runs Sidebar**: list of all the runs in your project.

* **Dot menu**: hover over a row in the sidebar to see the menu appear on the left side. Use this menu to rename a run, delete a run, or stop and active run.
* **Visibility icon**: click the eye to turn on and off runs on graphs
* **Color**: change the run color to another one of our presets or a custom color
* **Search**: search runs by name. This also filters visible runs in the plots.
* **Filter**: use the sidebar filter to narrow down the set of runs visible
* **Group**: select a config column to dynamically group your runs, for example by architecture. Grouping makes plots show up with a line along the mean value, and a shaded region for the variance of points on the graph.
* **Sort**: pick a value to sort your runs by, for example runs with the lowest loss or highest accuracy. Sorting will affect which runs show up on the graphs.
* **Expand button**: expand the sidebar into the full table
* **Run count**: the number in parentheses at the top is the total number of runs in the project. The number (N visualized) is the number of runs that have the eye turned on and are available to be visualized in each plot. In the example below, the graphs are only showing the first 10 of 183 runs. Edit a graph to increase the max number of runs visible.

If you pin, hide, or change the order of columns in the [Runs tab](#runs-tab), the Runs sidebar reflects these customizations.

**Panels layout**: use this scratch space to explore results, add and remove charts, and compare versions of your models based on different metrics

[View a live example](https://app.wandb.ai/example-team/sweep-demo)

{{< img src="/images/app_ui/workspace_tab_example.png" alt="Project workspace" >}}


### Add a section of panels

Click the section dropdown menu and click "Add section" to create a new section for panels. You can rename sections, drag them to reorganize them, and expand and collapse sections.

Each section has options in the upper right corner:

* **Switch to custom layout**: The custom layout allows you to resize panels individually.
* **Switch to standard layout**: The standard layout lets you resize all panels in the section at once, and gives you pagination.
* **Add section**: Add a section above or below from the dropdown menu, or click the button at the bottom of the page to add a new section.
* **Rename section**: Change the title for your section.
* **Export section to report**: Save this section of panels to a new report.
* **Delete section**: Remove the whole section and all the charts. This can be undone with the undo button at the bottom of the page in the workspace bar.
* **Add panel**: Click the plus button to add a panel to the section.

{{< img src="/images/app_ui/add-section.gif" alt="Adding workspace section" >}}

### Move panels between sections

Drag and drop panels to reorder and organize into sections. You can also click the "Move" button in the upper right corner of a panel to select a section to move the panel to.

{{< img src="/images/app_ui/move-panel.gif" alt="Moving panels between sections" >}}

### Resize panels

* **Standard layout**: All panels maintain the same size, and there are pages of panels. You can resize the panels by clicking and dragging the lower right corner. Resize the section by clicking and dragging the lower right corner of the section.
* **Custom layout**: All panels are sized individually, and there are no pages.

{{< img src="/images/app_ui/resize-panel.gif" alt="Resizing panels" >}}

### Search for metrics

Use the search box in the workspace to filter down the panels. This search matches the panel titles, which are by default the name of the metrics visualized.

{{< img src="/images/app_ui/search_in_the_workspace.png" alt="Workspace search" >}}

## Runs tab
<!-- Keep this in sync with /guide/models/track/runs/_index.md -->
Use the Runs tab to filter, group, and sort your runs.

{{< img src="/images/runs/run-table-example.png" alt="Runs table" >}}

The proceeding tabs demonstrate some common actions you can take in the Runs tab.

{{< tabpane text=true >}}
   {{% tab header="Customize columns" %}}
The Runs tab shows details about runs in the project. It shows a large number of columns by default.

{{% alert %}}
When you customize the Runs tab, the customization is also reflected in the **Runs** selector of the [Workspace tab]({{< relref "#workspace-tab" >}}).
{{% /alert %}}

- To view all visible columns, scroll the page horizontally.
- To change the order of the columns, drag a column to the left or right.
- To pin a column, hover over the column name, click the action menu `...`. that appears, then click **Pin column**. Pinned columns appear near the left of the page, after the **Name** column. To unpin a pinned column, choose **Unpin column**.
- To hide a column, hover over the column name, click the action menu `...`. that appears, then click **Hide column**. To view all columns that are currently hidden, click **Columns**.
- To show, hide, pin, and unpin multiple columns at once, click **Columns**.
  - Click the name of a hidden column to unhide it.
  - Click the name of a visible column to hide it.
  - Click the pin icon next to a visible column to pin it.

   {{% /tab %}}

   {{% tab header="Sort" %}}
Sort all rows in a Table by the value in a given column. 

1. Hover your mouse over the column title. A kebab menu will appear (three vertical docs).
2. Select on the kebab menu (three vertical dots).
3. Choose **Sort Asc** or **Sort Desc** to sort the rows in ascending or descending order, respectively. 

{{< img src="/images/data_vis/data_vis_sort_kebob.png" alt="Confident predictions" >}}

The preceding image demonstrates how to view sorting options for a Table column called `val_acc`.   
   {{% /tab %}}
   {{% tab header="Filter" %}}
Filter all rows by an expression with the **Filter** button on the top left of the dashboard. 

{{< img src="/images/data_vis/filter.png" alt="Incorrect predictions filter" >}}

Select **Add filter** to add one or more filters to your rows. Three dropdown menus will appear. From left to right the filter types are based on: Column name, Operator , and Values

|                   | Column name | Binary relation    | Value       |
| -----------       | ----------- | ----------- | ----------- |
| Accepted values   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |


The expression editor shows a list of options for each term using autocomplete on column names and logical predicate structure. You can connect multiple logical predicates into one expression using "and" or "or" (and sometimes parentheses).

{{< img src="/images/data_vis/filter_example.png" alt="Filtering runs by validation loss" >}}
The preceding image shows a filter that is based on the `val_loss` column. The filter shows runs with a validation loss less than or equal to 1.   
   {{% /tab %}}
   {{% tab header="Group" %}}
Group all rows by the value in a particular column with the **Group by** button in a column header. 

{{< img src="/images/data_vis/group.png" alt="Error distribution analysis" >}}

By default, this turns other numeric columns into histograms showing the distribution of values for that column across the group. Grouping is helpful for understanding higher-level patterns in your data.   
   {{% /tab %}}
{{< /tabpane >}}


<!-- ## Automations tab -->

## Automations tab
Automate downstream actions for versioning artifacts. To create an automation, define trigger events and resulting actions. Actions include executing a webhook or launching a W&B job. For more information, see [Automations]({{< relref "/guides/core/automations/" >}}).

{{< img src="/images/app_ui/automations_tab.png" alt="Automation tab" >}}

## Reports tab

See all the snapshots of results in one place, and share findings with your team.

{{< img src="/images/app_ui/reports-tab.png" alt="Reports tab" >}}

## Sweeps tab

Start a new [sweep]({{< relref "/guides/models/sweeps/" >}}) from your project.

{{< img src="/images/app_ui/sweeps-tab.png" alt="Sweeps tab" >}}

## Artifacts tab

View all [artifacts]({{< relref "/guides/core/artifacts/" >}}) associated with a project, from training datasets and [fine-tuned models]({{< relref "/guides/core/registry/" >}}) to [tables of metrics and media]({{< relref "/guides/models/tables/tables-walkthrough.md" >}}).

### Overview panel

{{< img src="/images/app_ui/overview_panel.png" alt="Artifact overview panel" >}}

On the overview panel, you'll find a variety of high-level information about the artifact, including its name and version, the hash digest used to detect changes and prevent duplication, the creation date, and any aliases. You can add or remove aliases here, take notes on both the version as well as the artifact as a whole.

### Metadata panel

{{< img src="/images/app_ui/metadata_panel.png" alt="Artifact metadata panel" >}}

The metadata panel provides access to the artifact's metadata, which is provided when the artifact is constructed. This metadata might include configuration arguments required to reconstruct the artifact, URLs where more information can be found, or metrics produced during the run which logged the artifact. Additionally, you can see the configuration for the run which produced the artifact as well as the history metrics at the time of logging the artifact.

### Usage panel

{{< img src="/images/app_ui/usage_panel.png" alt="Artifact usage panel" >}}

The Usage panel provides a code snippet for downloading the artifact for use outside of the web app, for example on a local machine. This section also indicates and links to the run which output the artifact and any runs which use the artifact as an input.

### Files panel

{{< img src="/images/app_ui/files_panel.png" alt="Artifact files panel" >}}

The files panel lists the files and folders associated with the artifact. W&B uploads certain files for a run automatically. For example, `requirements.txt` shows the versions of each library the run used, and `wandb-metadata.json`, and `wandb-summary.json` include information about the run. Other files may be uploaded, such as artifacts or media, depending on the run's configuration. You can navigate through this file tree and view the contents directly in the W&B web app.

[Tables]({{< relref "/guides/models/tables//tables-walkthrough.md" >}}) associated with artifacts are particularly rich and interactive in this context. Learn more about using Tables with Artifacts [here]({{< relref "/guides/models/tables//visualize-tables.md" >}}).

{{< img src="/images/app_ui/files_panel_table.png" alt="Artifact table view" >}}

### Lineage panel

{{< img src="/images/app_ui/lineage_panel.png" alt="Artifact lineage" >}}

The lineage panel provides a view of all of the artifacts associated with a project and the runs that connect them to each other. It shows run types as blocks and artifacts as circles, with arrows to indicate when a run of a given type consumes or produces an artifact of a given type. The type of the particular artifact selected in the left-hand column is highlighted.

Click the Explode toggle to view all of the individual artifact versions and the specific runs that connect them.

### Action History Audit tab

{{< img src="/images/app_ui/action_history_audit_tab_1.png" alt="Action history audit" >}}

{{< img src="/images/app_ui/action_history_audit_tab_2.png" alt="Action history" >}}

The action history audit tab shows all of the alias actions and membership changes for a Collection so you can audit the entire evolution of the resource.

### Versions tab

{{< img src="/images/app_ui/versions_tab.png" alt="Artifact versions tab" >}}

The versions tab shows all versions of the artifact as well as columns for each of the numeric values of the Run History at the time of logging the version. This allows you to compare performance and quickly identify versions of interest.

## Create a project
You can create a project in the W&B App or programmatically by specifying a project in a call to `wandb.init()`.

{{< tabpane text=true >}}
   {{% tab header="W&B App" %}}
In the W&B App, you can create a project from the **Projects** page or from a team's landing page.

From the **Projects** page:
1. Click the global navigation icon in the upper left. The navigation sidebar opens.
1. In the **Projects** section of the navigation, click **View all** to open the project overview page.
1. Click **Create new project**.
1. Set **Team** to the name of the team that will own the project.
1. Specify a name for your project using the **Name** field. 
1. Set **Project visibility**, which defaults to **Team**.
1. Optionally, provide a **Description**.
1. Click **Create project**.

From a team's landing page:
1. Click the global navigation icon in the upper left. The navigation sidebar opens.
1. In the **Teams** section of the navigation, click the name of a team to open its landing page.
1. In the landing page, click **Create new project**.
1. **Team** is automatically set to the team that owns the landing page you were viewing. If necessary, change the team.
1. Specify a name for your project using the **Name** field. 
1. Set **Project visibility**, which defaults to **Team**.
1. Optionally, provide a **Description**.
1. Click **Create project**.

   {{% /tab %}}
   {{% tab header="Python SDK" %}}
To create a project programmatically, specify a `project` when calling `wandb.init()`. If the project does not yet exist, it is created automatically, and is owned by the specified entity. For example:

```python
import wandb with wandb.init(entity="<entity>", project="<project_name>") as run: run.log({"accuracy": .95})
```

Refer to the [`wandb.init()` API reference]({{< relref "/ref/python/sdk/functions/init/#examples" >}}).
   {{% /tab %}}  
{{< /tabpane >}}

## Star a project

Add a star to a project to mark that project as important. Projects that you and your team mark as important with stars appear at the top of your organization's homepage.


For example, the proceeding image shows two projects that are marked as important, the `zoo_experiment` and `registry_demo`. Both projects appear within the top of the organization's homepage within the **Starred projects** section.
{{< img src="/images/track/star-projects.png" alt="Starred projects section" >}}


There are two ways to mark a project as important: within a project's overview tab or within your team's profile page.

{{< tabpane text=true >}}
    {{% tab header="Project overview" %}}
1. Navigate to your W&B project on the W&B App at `https://wandb.ai/<team>/<project-name>`.
2. Select the **Overview** tab from the project sidebar.
3. Choose the star icon in the upper right corner next to the **Edit** button.

{{< img src="/images/track/star-project-overview-tab.png" alt="Star project from overview" >}}    
    {{% /tab %}}
    {{% tab header="Team profile" %}}
1. Navigate to your team's profile page at `https://wandb.ai/<team>/projects`.
2. Select the **Projects** tab.
3. Hover your mouse next to the project you want to star. Click on star icon that appears.

For example, the proceeding image shows the star icon next to the "Compare_Zoo_Models" project.
{{< img src="/images/track/star-project-team-profile-page.png" alt="Star project from team page" >}}    
    {{% /tab %}}
{{< /tabpane >}}





Confirm that your project appears on the landing page of your organization by clicking on the organization name in the top left corner of the app.


## Delete a project

You can delete your project by clicking the three dots on the right of the overview tab.

{{< img src="/images/app_ui/howto_delete_project.gif" alt="Delete project workflow" >}}

If the project is empty, you can delete it by clicking the dropdown menu in the top-right and selecting **Delete project**.

{{< img src="/images/app_ui/howto_delete_project_2.png" alt="Delete empty project" >}}



## Add notes to a project

Add notes to your project either as a description overview or as a markdown panel within your workspace.

### Add description overview to a project

Descriptions you add to your page appear in the **Overview** tab of your profile.

1. Navigate to your W&B project
2. Select the **Overview** tab from the project sidebar
3. Choose Edit in the upper right hand corner
4. Add your notes in the **Description** field
5. Select the **Save** button

{{% alert title="Create reports to create descriptive notes comparing runs" %}}
You can also create a W&B Report to add plots and markdown side by side. Use different sections to show different runs, and tell a story about what you worked on.
{{% /alert %}}


### Add notes to run workspace

1. Navigate to your W&B project
2. Select the **Workspace** tab from the project sidebar
3. Choose the **Add panels** button from the top right corner
4. Select the **TEXT AND CODE** dropdown from the modal that appears
5. Select **Markdown**
6. Add your notes in the markdown panel that appears in your workspace