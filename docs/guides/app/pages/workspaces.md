---
description: A playground for exploring run data with interactive visualizations
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Workspaces

W&B workspace is your personal sandbox to customize charts and explore model results. A W&B workspace consists of *Tables* and *Panel sections*: 

* **Tables**: All runs logged to your project are listed in the project's table. Turn on and off runs, change colors, and expand the table to see notes, config, and summary metrics for each run.
* **Panel sections**: A section that contains one or more [panels](../features/panels/intro.md). Create new panels, organize them, and export to reports to save snapshots of your workspace.

![](/images/app_ui/workspace_table_and_panels.png)

## Workspace types
There are two main workspace categories: **Personal workspaces** and **Saved views**. 

* **Personal workspaces:**  A customizable workspace for in-depth analysis of models and data visualizations. Only the owner of the workspace can edit and save changes. Teammates can view a personal workspace but teammates can not make changes to someone else's personal workspace. 
* **Saved views:** Saved views are collaborative snapshots of a workspace. Anyone on your team can view, edit, and save changes to saved workspace views. Use saved workspace views for reviewing and discussing experiments, runs, and more.

The proceeding image shows multiple personal workspaces created by Cécile-parker's teammates. In this project, there are no saved views:
![](/images/app_ui/Menu_No_views.jpg)

## Saved workspace views
Improve team collaboration with tailored workspace views. Create Saved Views to organize your preferred setup of charts and data. 

### Create a new saved workspace view

1. Navigate to a personal workspace or a saved view.
2. Make edits to the workspace.
3. Click on the meatball menu (three horizontal dots) at the top right corner of your workspace. Click on **Save as a new view**.

New saved views appear in the workspace navigation menu.

![](/images/app_ui/Menu_Views.jpg)



### Update a saved workspace view 
Saved changes overwrite the previous state of the saved view. Unsaved changes are not retained. To update a saved workspace view in W&B:

1. Navigate to a saved view.
2. Make the desired changes to your charts and data within the workspace.
3. Click the **Save** button to confirm your changes. 

:::info
A confirmation dialog appears when you save your updates to a workspace view. If you prefer not to see this prompt in the future, select the option **Do not show this modal next time** before confirming your save.
:::

### Delete a saved workspace view
Remove saved views that are no longer needed.

1. Navigate to the saved view you want to remove.
2. Click on the hamburger menu (three horizontal lines) at the top right of the view.
3. Choose **Delete view**.
4. Confirm the deletion to remove the view from your workspace menu.

![](/images/app_ui/Deleting.gif)

### Share a workspace view
Share your customized workspace with your team by sharing the workspace URL directly. All users with access to the workspace project can see the saved Views of that workspace.


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


