---
description: A playground for exploring run data with interactive visualizations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Workspaces

W&B workspace is your personal sandbox to customize charts and explore model results. Workspaces consist of two main components: 

1. **Table**: All runs logged to your project are listed in the project's table. Turn on and off runs, change colors, and expand the table to see notes, config, and summary metrics for each run.
2. **Panel section**: [Panels](../features/panels/intro.md) are organized into panel sections. Create new panels, organize them, and export to reports to save snapshots of your workspace.

![](/images/app_ui/workspace_table_and_panels.png)



## Default workspace
Each W&B projects has a *default workspace*. The default workspace shows the custom workspace that was saved by the owner of that project. 

Each W&B user has one workspace that is unique to them and can be customized.

:::tip
Use default workspaces to set up a landing page for your public project, or help your team members get started.
:::

Changes that you make to your project workspace are saved automatically.  You can undo changes you make to your workspace with the undo buttons located at the bottom right of your workspace.

![](/images/app_ui/undo_button.png)

## View and customize workspaces
View and temporarily customize workspaces created by other W&B users. This is particularly useful if you want to conduct exploratory analysis of someone's work and possibly create a W&B report from it to share with others.

Changes you make to another user's workspace do not override the default workspace created by the owning W&B user.

You can undo temporary changes you make to a workspace you do not own with the **My Workspace** button located at the bottom of the W&B App UI:

1. Click on **My Workspace**
2. Select **Clear workspace**

![](/images/app_ui/workspaces_bar2.png)


## Team projects

Every user of a team will get one workspace that is unique to them and can be customized to their liking. However, users between teams can switch workspaces to other users of the team. Workspaces can differ between users for a variety of reasons like having different custom charts, different filters/groupings or section orders.

![](/images/app_ui/team_project_1.png)

You can fork a team member's workspace and then save it to your own. To fork a team member's workspace, click on the **Copy to My Workspace** button:

![](/images/app_ui/team_project_2.png)


## Sort charts into sections

You can sort charts into sections in your workspace programmatically or interactively with the W&B App UI.


<Tabs
  defaultValue="programmatically"
  values={[
    {label: 'Programmatically', value: 'programmatically'},
    {label: 'W&B App UI', value: 'ui'},
  ]}>
  <TabItem value="programmatically">

Add a prefix to the name of your metric when you log that metric to sort the chart into sections.

For example, the proceeding code block will produce two chart sections called **section-a** and **section-b**:

```python
run = wandb.init()
with run:
    for idx in range(100):
        run.log({"section-a/metric": idx})
        run.log({"section-b/metric": idx * 2})
```
![](/images/app_ui/workspaces_bar1.png)

  </TabItem>
  <TabItem value="ui">

1. Navigate to your project workspace.
2. Scroll down to the bottom of the panel section of your workspace.
3. Click on the **Add section** button to add a new section.

![](/images/app_ui/add_section_app.png)

  </TabItem>
</Tabs>



## Create multiple workspace versions (Beta)
Save custom versions of a workspace that belongs to you or someone on your team. The default workspace created by the owner of the project is tagged with a **Primary** label. 


### Create a new workspace version
A new version of a workspace is automatically created for you when you land on a project workspace that you do not own. This workspace version branches off the primary, default workspace version saved by the entity that owns the project. In other words, by default, you work on a workspace that mirrors the original workspace.

For example, in the following image a user renamed the automatically created workspace that branched off the **Primary** workspace to **DataScience-Member-initial-exploration**:

![](/images/app_ui/workspace_versions_initial_branched.png)


You can manually create a new version that is based off of [INSERT]. 

1. Navigate to the project workspace
2. Click on the name of the workspace
3. From the dropdown, select **New workspace**

![](/images/app_ui/create_manual_version.png)


### View workspace versions
Click on the name of the workspace to view workspace versions. A dropdown will appear that shows workspace versions and their lineage. Each workspace version created contains information of the workspace version that it was created from, including:

* The original workspace that the current workspace version is based off of
* When the branched version of the workspace was created
* The W&B user who created a workspace version



### Share your workspace version
Share your customized workspace with your team. 

1. Navigate to the project workspace
2. Click on the name of the workspace
3. Select the workspace version you want to share
4. Copy the URL 

Once you have the URL of the workspace version, share that URL to your team. 


<!-- ### Delete workspace versions -->