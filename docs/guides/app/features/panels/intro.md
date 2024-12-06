---
slug: /guides/app/features/panels
displayed_sidebar: default
title: Panels
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use workspace panel visualizations to explore your logged data, the relationships between hyperparameters and output metrics, and more. 

## Workspace modes

Weights and Biases projects support two different workspace modes: 

- **Automated workspaces** (default) automatically generate panels for all keys logged in the project. This can help you get started by visualizing all available data for the project.
- **Manual workspaces** start as a blank slate and display only those panels intentionally added by users. Choose a manual workspace when you  care mainly about a fraction of the keys logged in the project, or for a more focused analysis.

The icon next to the name of your workspace indicates how it generates panels:

| Icon | workspace mode |
| ----- | ----- |
| TBD | Automated |
| TBD | Manual |

To change how a workspace generates panels, [reset the workspace](#reset-a-workspace).

## Reset a workspace

To change a workspace's mode:

1. At the top of the workspace, click the `...` menu, then click **Reset workspace**.
2. Specify the [workspace mode](#workspace-modes), either **Automated** or **Manual**.
3. To save your changes, click either **Generate automated workspace** or **Get started**.

## Add panels

You can add panels to your workspace, either globally or at the section level.

To add a panel:

1. To add a panel directly to a section, click the section's `...` menu, then click **+ Add panels**.
2. To add a panel to the top level (for manual workspaces only), click **+ Add panels** at the top of the workspace.
3. Select the type of panel to add.
![](/images/app_ui/add_single_panel.gif)

### Quick add

The **Quick Add** feature allows you to generate a standard panel for any key logged in the project by selecting the key from the list using the search field to find a specific key or path.

In an [automated workspace](#workspace-modes), **Quick Add** is available only if you previously deleted panels.

### Custom add

To add a custom panel to your workspace:

1. Select the type of panel you’d like to create.
2. Follow the prompts to configure the panel.

To learn more about the options for each type of panel, refer to the relevant section below, such as [Line plots](line-plot/intro.md) or [Bar plots](bar-plot.md).

Learn more about the types of panels you can create:

- Line plots
- Bar plots
- Parallel coordinates
- Scatter plots
- Save and diff code
- Parameter importance


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

## Manage panels

### Edit a panel

To edit a panel:

1. Click its pencil icon.
2. Modify the panel's settings.
3. To change the panel to a different type, select the type and then configure the settings.
4. Click **Apply**.

### Move a panel

To move a panel:

1. Click the `...` menu for the panel.
2. Click **Move**.
3. If the workspace generates panels automatically, you must select a new section for the panel. If the workspace generates panels manually, you can select a new section or move the panel to the top level of the workspace.

## Duplicate a panel

To duplicate a panel:

1. At the top of the panel, click the `...` menu.
2. Click **Duplicate**.

If desired, you can customize or [move](#move-a-panel) the duplicated panel.

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
1. To add a section, click **Add section**. To add a new section above or below an existing section, you can instead click the section's `...` menu, then click **New section below** or **New section above**.
1. To delete a section, click its `...` menu, then click **Delete section**. This removes the section and its panels.


