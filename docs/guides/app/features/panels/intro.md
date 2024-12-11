---
slug: /guides/app/features/panels
displayed_sidebar: default
title: Panels
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use workspace panel visualizations to explore your [logged data](/ref/python/log.md) by key, visualize the relationships between hyperparameters and output metrics, and more. 

## Workspace modes

W&B projects support two different workspace modes. The icon next to the workspace name shows its mode. 

| Icon | Workspace mode |
| --- | --- |
| <img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" width="32px" /> | **Automated workspaces** automatically generate panels for all keys logged in the project. This can help you get started by visualizing all available data for the project. |
| <img src="/images/app_ui/manual_workspace.svg" alt="manual workspace icon" width="32px" /> | **Manual workspaces** start as blank slates and display only those panels intentionally added by users. Choose a manual workspace when you care mainly about a fraction of the keys logged in the project, or for a more focused analysis. |

To change how a workspace generates panels, [reset the workspace](#reset-a-workspace).

:::tip Undo changes to your workspace
To undo changes to your workspace, click the Undo button (arrow that points left) or type **CMD + Z** (macOS) or **CTRL + Z** (Windows / Linux).
:::

## Reset a workspace

To reset a workspace:

1. At the top of the workspace, click the action menu `...`.
1. Click **Reset workspace**.

## Add panels

You can add panels to your workspace, either globally or at the section level.

To add a panel:

1. To add a panel globally, click **Add panels** in the control bar near the panel search field.
1. To add a panel directly to a section instead, click the section's action `...` menu, then click **+ Add panels**.
1. Select the type of panel to add.
![](/images/app_ui/add_single_panel.gif)

### Quick add

**Quick Add** allows you to select a key in the project from a list to generate a standard panel for it.

For an automated workspace with no deleted panels, **Quick add** is not available. You can use **Quick add** to re-add a panel that you deleted.

### Custom panel add

To add a custom panel to your workspace:

1. Select the type of panel you’d like to create.
1. Follow the prompts to configure the panel.

To learn more about the options for each type of panel, refer to the relevant section below, such as [Line plots](line-plot/intro.md) or [Bar plots](bar-plot.md).

## Manage panels

### Edit a panel

To edit a panel:

1. Click its pencil icon.
1. Modify the panel's settings.
1. To change the panel to a different type, select the type and then configure the settings.
1. Click **Apply**.

### Move a panel

To move a panel to a different section, you can use the drag handle on the panel. To select the new section from a list instead:

1. If necessary, create a new section by clicking **Add section** after the last section.
1. Click the  action `...` menu for the panel.
1. Click **Move**, then select a new section.

You can also use the drag handle to rearrange panels within a section.

## Duplicate a panel

To duplicate a panel:

1. At the top of the panel, click the action `...` menu.
1. Click **Duplicate**.

If desired, you can [customize](#edit-a-panel) or [move](#move-a-panel) the duplicated panel.

### Remove panels

To remove a panel:

1. Hover your mouse over the panel.
1. Select the action `...` menu.
1. Click **Delete**.

To remove all panels from a manual workspace, click its action `...` menu, then click **Clear all panels**.

To remove all panels from an automatic or manual workspace, you can [reset the workspace](#reset-a-workspace). Select **Automatic** to start with the default set of panels, or select **Manual** to start with an empty workspace with no panels.

## Manage sections

By default, sections in a workspace reflect the logging hierarchy of your keys. However, in a manual workspace, sections appear only after you start adding panels.

### Add a section

To add a section, click **Add section** after the last section.

To add a new section before or after an existing section, you can instead click the section's action `...` menu, then click **New section below** or **New section above**.

### Rename a section

To rename a section, click its action `...` menu, then click **Rename section**.

### Delete a section

To delete a section, click its `...` menu, then click **Delete section**. This removes the section and its panels.


