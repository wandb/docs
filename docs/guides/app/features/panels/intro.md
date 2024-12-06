---
slug: /guides/app/features/panels
displayed_sidebar: default
title: Panels
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use panel visualizations to explore your logged data, the relationships between hyperparameters and output metrics, and more. 

## Check workspace settings

The icon next to the name of your workspace indicates how it generates panels:

- A green diamond indicates that the workspace generates panels automatically.
- A purple safety pin and ruler indicates that the workspace generates panels manually.

To configure panel generation, [reset the workspace](#reset-a-workspace).

## Reset a workspace

By default, a workspace automatically generates panels for all keys [`log`](../../../../ref/python/log.md) in the project. This can help you get started quickly by visualizing all available data for the project.

:::info
When a workspace has automatic panel generation enabled, W&B uses the key value you specify with [`log`](../../../../ref/python/log.md) to determine whether or not to create a new panel. 
:::


If desired, you can configure the workspace to display only those panels you add manually. Resetting a workspace removes all custom sections and panels. Resetting an automatic workspace adds back panels that were previously removed.

To change the type for a workspace:

1. At the top of the workspace, click the `...` menu, then click **Reset workspace**.
2. To generate panels automatically, select **Automatic**, then click **Generate automated workspace**.
3. To generate panels manually instead, select **Manual**, then click **Get started**.

## Add panels

To add a panel:

1. To add a panel directly to a section, click the section's `...` menu, then click **+ Add panels**.
2. To add a panel to the top level (for manual workspaces only) or to an arbitrary section, click **+ Add panels** at the top of the workspace.
3. From the dropdown, select the type of panel to add.
![](/images/app_ui/add_single_panel.gif)
4. (Optional) If prompted, define parameters for the panel. 

<Tabs
  defaultValue="quick"
  values={[
    {label: 'Add a plot from a logged value', value: 'quick'},
    {label: 'Add a custom plot', value: 'single'},
  ]}>
  <TabItem value="quick">

1. Click **Add panels**.
2. Click **Quick add**.
2. Provide a regular expression within the search field or select a key from the **KEYS** dropdown.

  </TabItem>
  <TabItem value="single">

1. Click **Add panels**.
2. From the dropdown, select the type of panel to add.
3. (Optional) If prompted, define parameters for the panel. 
3. Select **Apply**.

  </TabItem> 
</Tabs>


:::tip Undo changes to your workspace
Select the undo button (arrow that points left) to undo any unwanted changes.
:::

You can add up to 500 panels at a time. To add multiple panels:

1. Follow the steps to [add a panel](#add-panels) to a section or the top level of the workspace, but choose **Quick add**.
2. Provide a regular expression within the search field.
3. Click **Add all**.
![](/images/app_ui/bulk_panels.gif)

:::note
The **Add all** button appears only if a regular expression match occurs.
:::

## Duplicate a panel

To duplicate a panel:

1. At the top of the panel, click the `...` menu.
2. Click **Duplicate**.
3. If desired, customize the duplicate panel.
4. If necessary, you can [move the duplicate panel](#move-a-panel).

## Manage panels

### Move a panel

To move a panel:

1. Click the `...` menu for the panel.
2. Click **Move**.
3. If the workspace generates panels automatically, you must select a new section for the panel. If the workspace generates panels manually, you can select a new section or move the panel to the top level of the workspace.

### Edit a panel

To edit a panel:

1. Click its pencil icon.
2. Modify the panel's settings.
3. To change the panel to a different type, select the type and then configure the settings.
4. Click **Apply**.

### Remove panels

To remove a panel:

1. Hover your mouse in the upper corner of the panel you want to remove
2. Select the three horizontal dots (**...**) that appear
3. From the dropdown, select **Delete**

To remove all panels from a manual workspace, click its `...` menu, then click **Clear all panels**.

To remove all panels from an automatic or manual workspace, you can [reset the workspace](#reset-a-workspace). Select **Automatic** to start with the default set of panels, or select **Manual** to start with an empty workspace with no panels.

## Manage sections

Sections help you keep your workspace organized so you can focus on your most important data, experiments, and visualizations. You can create new sections, and you can [move a panel](#move-a-panel) from one section to another.

By default, an automated workspace adds panels to sections according to their type, and shows panels in **Chart** and **System** sections by default. When you add a panel to an automated workspace, you must choose its section.

In an automatic workspace, you must choose a section when adding a new panel. You can [move a panel](#move-a-panel) to a different section, but not to the top level of the workspace.

By default, a manual workspace has no sections. You can optionally create panels in sections or at the top level of the workspace.

1. To expand or collapse all sections, click the `...` menu next to the panel search field, then select **Expand all sections** or **Collapse all sections**.
1. To add a section, click **Add section**. The new section has a default name, such as **Panel Section 0**. To rename a section, click its  `...` menu.
1. To delete a section, click its `...` menu, then click **Delete section**. This removes the section and its panels.


