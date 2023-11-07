---
description: A playground for exploring run data with interactive visualizations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Workspaces

Your workspace is your personal sandbox to customize charts and explore model results.

1. **Table**: All the runs in your project are listed in the table. Turn on and off runs, change colors, and expand the table to see notes, config, and summary metrics for each run.
2. **Panels**: Panels are organized into sections. Create new panels, organize them, and export to reports to save snapshots of your workspace.

![](/images/app_ui/workspace_table_and_panels.png)



## Default workspace
Each W&B projects has a *default workspace*. The default workspace shows the custom workspace that was saved by the owner of that project. Only the owner of that project can edit the default workspace. Each W&B user has one workspace that is unique to them and can be customized.

:::tip
Use default workspaces to set up a landing page for your public project, or help your team members get started.
:::


At the bottom of a project page there is a button labeled **My Workspace**. Click on My Workspace button to clear 


Changes that you make to your project workspace are saved automatically.  You can undo those changes with the undo button located at the bottom right of your workspace.

## View and customize workspaces
View and temporarily customize workspaces created by other W&B users. This is particularly useful if you want to conduct exploratory analysis of someone's work and possibly create a W&B report from it to share with others.

Changes you make to another user's workspace do not override the default workspace created by the owning W&B user.

Undo the temporary changes you make to a workspace you do not own with the **My Workspace** button located at the bottom of the W&B App UI:

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


  </TabItem>
</Tabs>



## Create multiple workspaces for a project (Public Preview) 