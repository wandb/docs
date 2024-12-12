---
description: >-
  Compare versions of your model, explore results in a scratch workspace, and
  export findings to a report to save notes and visualizations
title: Projects
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


A *project* is a central location where you visualize results, compare experiments, view and download artifacts, create an automation, and more. 

{{% alert %}}
Each project has a visibility setting that determines who can access it. For more information about who can access a project, see [Project visibility](../hosting/iam/restricted-projects.md).
{{% /alert %}}

Each project contains the proceeding which you can access from the sidebar:

* [**Overview**](project-page.md#overview-tab): snapshot of your project
* [**Workspace**](project-page.md#workspace-tab): personal visualization sandbox
* [**Runs**](#runs-tab): A table that lists all the runs in your project
* **Automations**: Automations configured in your project
* [**Sweeps**](project-page.md#sweeps-tab): automated exploration and optimization
* [**Reports**](project-page.md#reports-tab): saved snapshots of notes, runs, and graphs
* [**Artifacts**](#artifacts-tab): Contains all runs and the artifacts associated with that run

## Overview tab

* **Project name**: The name of the project. W&B creates a project for you when you initialize a run with the name you provide for the project field. You can change the name of the project at any time by selecting the **Edit** button in the upper right corner.
* **Description**: A description of the project.
* **Project visibility**: The visibility of the project. The visibility setting that determines who can access it. See [Project visibility](../hosting/iam/restricted-projects.md) for more information.
* **Last active**: Timestamp of the last time data is logged to this project
* **Owner**: The entity that owns this project
* **Contributors**: The number of users that contribute to this project
* **Total runs**: The total number of runs in this project
* **Total compute**: we add up all the run times in your project to get this total
* **Undelete runs**: Click the dropdown menu and click "Undelete all runs" to recover deleted runs in your project.
* **Delete project**: click the dot menu in the right corner to delete a project

[View a live example](https://app.wandb.ai/example-team/sweep-demo/overview)

![](/images/track/overview_tab_image.png)


## Workspace tab

A project's *workspace* gives you a personal sandbox to compare experiments. Use projects to organize models that can be compared, working on the same problem with different architectures, hyperparameters, datasets, preprocessing etc.


**Runs Sidebar**: list of all the runs in your project

* **Dot menu**: hover over a row in the sidebar to see the menu appear on the left side. Use this menu to rename a run, delete a run, or stop and active run.
* **Visibility icon**: click the eye to turn on and off runs on graphs
* **Color**: change the run color to another one of our presets or a custom color
* **Search**: search runs by name. This also filters visible runs in the plots.
* **Filter**: use the sidebar filter to narrow down the set of runs visible
* **Group**: select a config column to dynamically group your runs, for example by architecture. Grouping makes plots show up with a line along the mean value, and a shaded region for the variance of points on the graph.
* **Sort**: pick a value to sort your runs by, for example runs with the lowest loss or highest accuracy. Sorting will affect which runs show up on the graphs.
* **Expand button**: expand the sidebar into the full table
* **Run count**: the number in parentheses at the top is the total number of runs in the project. The number (N visualized) is the number of runs that have the eye turned on and are available to be visualized in each plot. In the example below, the graphs are only showing the first 10 of 183 runs. Edit a graph to increase the max number of runs visible.

**Panels layout**: use this scratch space to explore results, add and remove charts, and compare versions of your models based on different metrics

[View a live example](https://app.wandb.ai/example-team/sweep-demo)

![](/images/app_ui/workspace_tab_example.png)


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

![](/images/app_ui/add-section.gif)

### Move panels between sections

Drag and drop panels to reorder and organize into sections. You can also click the "Move" button in the upper right corner of a panel to select a section to move the panel to.

![](/images/app_ui/move-panel.gif)

### Resize panels

* **Standard layout**: All panels maintain the same size, and there are pages of panels. You can resize the panels by clicking and dragging the lower right corner. Resize the section by clicking and dragging the lower right corner of the section.
* **Custom layout**: All panels are sized individually, and there are no pages.

![](/images/app_ui/resize-panel.gif)

### Search for metrics

Use the search box in the workspace to filter down the panels. This search matches the panel titles, which are by default the name of the metrics visualized.

![](/images/app_ui/search_in_the_workspace.png)

<!-- ## Table Tab

Use the table to filter, group, and sort your results.

[View a live example](https://app.wandb.ai/example-team/sweep-demo/table?workspace=user-carey) -->




<!-- start -->


## Runs tab
Use the runs tab to filter, group, and sort your results.

![](/images/runs/run-table-example.png)

<!-- [Try these yourself →](https://wandb.ai/stacey/mnist-viz/artifacts/predictions/baseline/d888bc05719667811b23/files/predictions.table.json) -->

The proceeding tabs demonstrate some common actions you can take in the runs tab.

<Tabs
  defaultValue="sort"
  values={[
    {label: 'Sort', value: 'sort'},
    {label: 'Filter', value: 'filter'},
    {label: 'Group', value: 'group'},
  ]}>
  <TabItem value="sort">

Sort all rows in a Table by the value in a given column. 

1. Hover your mouse over the column title. A kebob menu will appear (three vertical docs).
2. Select on the kebob menu (three vertical dots).
3. Choose **Sort Asc** or **Sort Desc** to sort the rows in ascending or descending order, respectively. 

![See the digits for which the model most confidently guessed "0".](/images/data_vis/data_vis_sort_kebob.png)

The preceding image demonstrates how to view sorting options for a Table column called `val_acc`.

</TabItem>
  <TabItem value="filter">
  
Filter all rows by an expression with the **Filter** button on the top left of the dashboard. 

![See only examples which the model gets wrong.](/images/data_vis/filter.png)

Select **Add filter** to add one or more filters to your rows. Three dropdown menus will appear. From left to right the filter types are based on: Column name, Operator , and Values

|                   | Column name | Binary relation    | Value       |
| -----------       | ----------- | ----------- | ----------- |
| Accepted values   | String       |  &equals;, &ne;, &le;, &ge;, IN, NOT IN,  | Integer, float, string, timestamp, null |


The expression editor shows a list of options for each term using autocomplete on column names and logical predicate structure. You can connect multiple logical predicates into one expression using "and" or "or" (and sometimes parentheses).

![](/images/data_vis/filter_example.png)
The preceding image shows a filter that is based on the `val_loss` column. The filter shows runs with a validation loss less than or equal to 1.

</TabItem>
  <TabItem value="group">

Group all rows by the value in a particular column with the **Group by** button in a column header. 

![The truth distribution shows small errors: 8s and 2s are confused for 7s and 9s for 2s.](/images/data_vis/group.png)

By default, this turns other numeric columns into histograms showing the distribution of values for that column across the group. Grouping is helpful for understanding higher-level patterns in your data.

  </TabItem>
</Tabs>


<!-- ## Automations tab -->


## Reports tab

See all the snapshots of results in one place, and share findings with your team.

![](/images/app_ui/reports-tab.png)

## Sweeps tab

Start a new [sweep](../sweeps/intro.md) from your project.

![](/images/app_ui/sweeps-tab.png)

## Artifacts tab

View all the [artifacts](../artifacts/intro.md) associated with a project, from training datasets and [fine-tuned models](../model_registry/intro.md) to [tables of metrics and media](../tables/tables-walkthrough.md).

### Overview panel

![](/images/app_ui/overview_panel.png)

On the overview panel, you'll find a variety of high-level information about the artifact, including its name and version, the hash digest used to detect changes and prevent duplication, the creation date, and any aliases. You can add or remove aliases here, take notes on both the version as well as the artifact as a whole.

### Metadata panel

![](/images/app_ui/metadata_panel.png)

The metadata panel provides access to the artifact's metadata, which is provided when the artifact is constructed. This metadata might include configuration arguments required to reconstruct the artifact, URLs where more information can be found, or metrics produced during the run which logged the artifact. Additionally, you can see the configuration for the run which produced the artifact as well as the history metrics at the time of logging the artifact.

### Usage panel

![](/images/app_ui/usage_panel.png)

The Usage panel provides a code snippet for downloading the artifact for use outside of the web app, for example on a local machine. This section also indicates and links to the run which output the artifact and any runs which use the artifact as an input.

### Files panel

![](/images/app_ui/files_panel.png)

The files panel lists the files and folders associated with the artifact. W&B uploads certain files for a run automatically. For example, `requirements.txt` shows the versions of each library the run used, and `wandb-metadata.json`, and `wandb-summary.json` include information about the run. Other files may be uploaded, such as artifacts or media, depending on the run's configuration. You can navigate through this file tree and view the contents directly in the W&B web app.

[Tables](../tables/tables-walkthrough.md) associated with artifacts are particularly rich and interactive in this context. Learn more about using Tables with Artifacts [here](../tables/visualize-tables.md).

![](/images/app_ui/files_panel_table.png)

### Lineage panel

![](/images/app_ui/lineage_panel.png)

The lineage panel provides a view of all of the artifacts associated with a project and the runs that connect them to each other. It shows run types as blocks and artifacts as circles, with arrows to indicate when a run of a given type consumes or produces an artifact of a given type. The type of the particular artifact selected in the left-hand column is highlighted.

Click the Explode toggle to view all of the individual artifact versions and the specific runs that connect them.

### Action History Audit tab

![](/images/app_ui/action_history_audit_tab_1.png)

![](/images/app_ui/action_history_audit_tab_2.png)

The action history audit tab shows all of the alias actions and membership changes for a Collection so you can audit the entire evolution of the resource.

### Versions tab

![](/images/app_ui/versions_tab.png)

The versions tab shows all versions of the artifact as well as columns for each of the numeric values of the Run History at the time of logging the version. This allows you to compare performance and quickly identify versions of interest.



## Star a project

Add a star to a project to mark that project as important. Projects that you and your team mark as important with stars appear at the top of your organization's home page.


For example, the proceeding image shows two projects that are marked as important, the "zoo_experiment" and "registry_demo". Both projects appear within the top of the organization's home page within the **Starred projects** section.
![](/images/track/star-projects.png)


There are two ways to mark a project as important: within a project's overview tab or within your team's profile page.


<Tabs
  defaultValue="project_overview"
  values={[
    {label: 'Project overview', value: 'project_overview'},
    {label: 'Team profile', value: 'project_landing_page'},
  ]}>
  <TabItem value="project_overview">

1. Navigate to your W&B project on the W&B App at `https://wandb.ai/<team>/<project-name>`.
2. Select the **Overview** tab from the project sidebar.
3. Choose the star icon in the upper right corner next to the **Edit** button.

![](/images/track/star-project-overview-tab.png)

  </TabItem>
  <TabItem value="project_landing_page">

1. Navigate to your team's profile page at `https://wandb.ai/<team>/projects`.
2. Select the **Projects** tab.
3. Hover your mouse next to the project you want to star. Click on star icon that appears.

For example, the proceeding image shows the star icon next to the "Compare_Zoo_Models" project.
![](/images/track/star-project-team-profile-page.png)

  </TabItem>
</Tabs>



Confirm that your project appears on the landing page of your organization by clicking on the organization name in the top left corner of the app.


## Delete a project

You can delete your project by clicking the three dots on the right of the overview tab.

![](/images/app_ui/howto_delete_project.gif)

If the project is empty, you can delete it by clicking the dropdown menu in the top-right and selecting "Delete project".

![](/images/app_ui/howto_delete_project_2.png)



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