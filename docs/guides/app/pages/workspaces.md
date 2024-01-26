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



## Create saved workspace views (Beta)
Improve team collaboration with tailored workspace views. 

### Understand the workspace categories

* **Personal Workspaces:** Exclusive to you, these are customizable spaces for in-depth analysis of models and data visualizations. You have edit control, while teammates can view configurations without altering them.

* **Saved Views:** These are collaborative snapshots of your workspace, viewable and usable by all project collaborators. They serve as fixed references of particular workspace states for collective review and discussion.
 
 ![](/images/app_ui/Menu_No_views.jpg)

### Create a new saved workspace view
A Workspace View in Weights & Biases lets you organize and save your preferred workspace setup of charts and data. You can easily create a new View by following these steps:

* **Open a Workspace or View:** Start by going to the Workspace or View you wish to save.
* **Save View:** Look for the meatball menu (three horizontal dots) at the top right corner of your workspace. Click on it and then choose **Save as a new view**. Once saved, you can also give your new view a descriptive name.
* **Find Your New View:** Once saved, new views appear in the workspace navigation menu. 

 ![](/images/app_ui/Menu_Views.jpg)


### Update a saved workspace view 
To update a saved workspace view in Weights & Biases:

* **Edit the Workspace:** Make the desired changes to your charts and data within the workspace.
* **Save the Changes:** Click the **Save** button to confirm your changes. Saved changes overwrite the previous state of the saved view. Unsaved changes are not retained.

:::info
A confirmation  dialog appears when you save your updates to a workspace view. If you prefer not to see this prompt in the future, simply check the option **Do not show this modal next time** before confirming your save.
:::

### Deleting a saved workspace view
To delete a view and manage your workspace menu in Weights & Biases:

* Navigate to the view you wish to remove.
* Click on the meatball menu (three horizontal lines) at the top right of the view.
* Choose the option to **Delete view**.
* Confirm the deletion to remove the view from your workspace menu.

This process helps to declutter your workspace by removing any views that are no longer needed.

 ![](/images/app_ui/Deleting.gif)

### Share a workspace view
Share your customized workspace with your team by sharing the workspace URL directly. All users with access to the workspace project can see the saved Views of that workspace.



