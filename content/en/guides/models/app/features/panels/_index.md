---
menu:
  default:
    identifier: intro_panels
    parent: w-b-app-ui-reference
title: Panels
weight: 1
url: guides/app/features/panels
cascade:
- url: guides/app/features/panels/:filename
---

Use workspace panel visualizations to explore your [logged data]({{< relref "/ref/python/sdk/run.md/#method-runlog" >}}) by key, visualize the relationships between hyperparameters and output metrics, and more. 

## Workspace modes

W&B projects support two different workspace modes. The icon next to the workspace name shows its mode. 

| Icon | Workspace mode |
| --- | --- |
| {{< img src="/images/app_ui/automated_workspace.svg" alt="automated workspace icon" width="32px" >}} | **Automated workspaces** automatically generate panels for all keys logged in the project. Choose an automatic workspace:<ul><li>To get started quickly by visualizing all available data for the project.</li><li>For a smaller projects that log fewer keys.</li><li>For more broad analysis.</li></ul>If you delete a panel from an automatic workspace, you can use [Quick add]({{< relref "#quick-add" >}}) to recreate it. |
| {{<img src="/images/app_ui/manual_workspace.svg" alt="manual workspace icon" width="32px" >}} | **Manual workspaces** start as blank slates and display only those panels intentionally added by users. Choose a manual workspace:<ul><li>When you care mainly about a fraction of the keys logged in the project.</li><li>For more focused analysis.</li><li>To improve the performance of a workspace, avoiding loading panels that are less useful to you.</li></ul>Use [Quick add]({{< relref "#quick-add" >}}) to easily populate a manual workspace and its sections with useful visualizations rapidly. |

To change how a workspace generates panels, [reset the workspace]({{< relref "#reset-a-workspace" >}}).

{{% alert title="Undo changes to your workspace" %}} 
To undo changes to your workspace, click the Undo button (arrow that points left) or type **CMD + Z** (macOS) or **CTRL + Z** (Windows / Linux).
{{% /alert %}}

## Reset a workspace

To reset a workspace:

1. At the top of the workspace, click the action menu `...`.
1. Click **Reset workspace**.

## Configure the workspace layout {#configure-workspace-layout}

To configure the workspace layout, click **Settings** near the top of the workspace, then click **Workspace layout**.

- **Hide empty sections during search** (turned on by default)
- **Sort panels alphabetically** (turned off by default)
- **Section organization** (grouped by first prefix by default). To modify this setting:
  1. Click the padlock icon.
  1. Choose how to group panels within a section.

To configure defaults for the workspace's line plots, refer to [Line plots]({{< relref "line-plot/#all-line-plots-in-a-workspace" >}}).

### Configure a section's layout {#configure-section-layout}

To configure the layout of a section, click its gear icon, then click **Display preferences**.
- **Turn on or off colored run names in tooltips** (turned on by default)
- **Only show highlighted run in companion chart tooltips** (turned off by default)
- **Number of runs shown in tooltips** (a single run, all runs, or **Default**)
- **Display full run names on the primary chart tooltip** (turned off by default)

## View a panel in full-screen mode

In full-screen mode, the run selector displays and panels use full-fidelity sampling mode plots with 10,000 buckets, rather than 1000 buckets otherwise.

To view a panel in full-screen mode:

1. Hover over the panel.
1. Click the panel's action menu `...`, then click the full-screen button, which looks like a viewfinder or an outline showing the four corners of a square.
    {{< img src="/images/app_ui/panel_fullscreen.png" alt="Full-screen panel" >}}
1. When you [share the panel]({{< relref "#share-a-panel" >}}) while viewing it in full-screen mode, the resulting link opens in full-screen mode automatically.

To get back to a panel's workspace from full-screen mode, click the left-pointing arrow at the top of the page. To navigate through a section's panels without exiting full-screen mode, use either the **Previous** and **Next** buttons below the panel or the left and right arrow keys.

## Add panels

This section shows various ways to add panels to your workspace.

### Add a panel manually

Add panels to your workspace one at a time, either globally or at the section level.

1. To add a panel globally, click **Add panels** in the control bar near the panel search field.
1. To add a panel directly to a section instead, click the section's action `...` menu, then click **+ Add panels**.
1. Select the type of panel to add, such as a chart. The panel's configuration details appear, with defaults selected.
1. Optionally, customize the panel and its display preferences. Configuration options depend on the type of panel you select. To learn more about the options for each type of panel, refer to the relevant section below, such as [Line plots]({{< relref "line-plot/" >}}) or [Bar plots]({{< relref "bar-plot.md" >}}).
1. Click **Apply**.

{{< img src="/images/app_ui/add_single_panel.gif" alt="Demo of adding a panel" >}} 

### Quick add panels {#quick-add}

Use **Quick add** to add a panel automatically for each key you select, either globally or at the section level.

{{% alert %}}
For an automated workspace with no deleted panels, the **Quick add** option is not visible because the workspace already includes panels for all logged keys. You can use **Quick add** to re-add a panel that you deleted.
{{% /alert %}}

1. To use **Quick add** to add a panel globally, click **Add panels** in the control bar near the panel search field, then click **Quick add**.
1. To use **Quick add** to add a panel directly to a section, click the section's action `...` menu, click **Add panels**, then click **Quick add**.
1. A list of panels appears. Each panel with a checkmark is already included in the workspace.
    - To add all available panels, click the **Add <N> panels** button at the top of the list. The **Quick Add** list closes and the new panels display in the workspace.
    - To add an individual panel from the list, hover over the panel's row, then click **Add**. Repeat this step for each panel you want to add, then click the **X** at the top right to close the **Quick Add** list. The new panels display in the workspace.
1. Optionally, customize the panel's settings.

## Share a panel

This section shows how to share a panel using a link.

To share a panel using a link, you can either:

- While viewing the panel in full-screen mode, copy the URL from the browser.
- Click the action menu `...` and select **Copy panel URL**.

Share the link with the user or team. When they access the link, the panel opens in [full-screen mode]({{< relref "#view-a-panel-in-full-screen-mode" >}}).

To return to a panel's workspace from full-screen mode, click the left-pointing arrow at the top of the page.

### Compose a panel's full-screen link programmatically
In certain situations, such as when [creating an automation]({{< relref "/guides/core/automations/" >}}), it can be useful to include the panel's full-screen URL. This section shows the format for a panel's full-screen URL. In the proceeding example, replace the entity, project, panel, and section names in brackets.

```text
https://wandb.ai/<ENTITY_NAME>/<PROJECT_NAME>?panelDisplayName=<PANEL_NAME>&panelSectionName=<SECTON_NAME>
```

If multiple panels in the same section have the same name, this URL opens the first panel with the name.

### Embed or share a panel on social media
To embed a panel in a website or share it on social media, the panel must be viewable by anyone with the link. If a project is private, only members of the project can view the panel. If the project is public, anyone with the link can view the panel.

To get the code to embed or share a panel on social media:

1. From the workspace, hover over the panel, then click its action menu `...`.
1. Click the **Share** tab.
1. Change **Only those who are invited have access** to **Anyone with the link can view**. Otherwise, the choices in the next step are not available.
1. Choose **Share on Twitter**, **Share on Reddit**, **Share on LinkedIn**, or **Copy embed link**.

### Email a panel report
To email a single panel as a stand-alone report:
1. Hover over the panel, then click the panel's action menu `...`.
1. Click **Share panel in report**.
1. Select the **Invite** tab.
1. Enter an email address or username.
1. Optionally, change **can view** to **can edit**.
1. Click **Invite**. W&B sends an email to the user with a clickable link to the report that contains only the panel you are sharing. 

Unlike when you [share a panel]({{< relref "#share-a-panel" >}}), the recipient cannot get to the workspace from this report.

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

### Duplicate a panel

To duplicate a panel:

1. At the top of the panel, click the action `...` menu.
1. Click **Duplicate**.

If desired, you can [customize]({{< relref "#edit-a-panel" >}}) or [move]({{< relref "#move-a-panel" >}}) the duplicated panel.

### Remove panels

To remove a panel:

1. Hover your mouse over the panel.
1. Select the action `...` menu.
1. Click **Delete**.

To remove all panels from a manual workspace, click its action `...` menu, then click **Clear all panels**.

To remove all panels from an automatic or manual workspace, you can [reset the workspace]({{< relref "#reset-a-workspace" >}}). Select **Automatic** to start with the default set of panels, or select **Manual** to start with an empty workspace with no panels.

## Manage sections

By default, sections in a workspace reflect the logging hierarchy of your keys. However, in a manual workspace, sections appear only after you start adding panels.

### Add a section

To add a section, click **Add section** after the last section.

To add a new section before or after an existing section, you can instead click the section's action `...` menu, then click **New section below** or **New section above**.


### Manage a section's panels
Sections with a large number of panels are paginated by default. The default number of panels on a page depend on the panel's configuration and on the sizes of the panels in the section.

{{% alert %}}
The **Custom grid** layout will soon be removed. W&B suggests that you no longer use Custom grid layouts. Consider updating your workspace from **Custom grid** to **Standard grid**.

When the **Custom grid** layout is removed, workspaces will be updated to use the **Standard grid** layout, which will no longer be configurable.
{{% /alert %}}

1. To check which layout a section uses, click the section's action `...` menu. To change a section's layout, select **Standard grid** or **Custom grid** in the **Layout grid** section.
1. To resize a panel, hover over it, click the drag handle, and drag it to adjust the panel's size.
  - If a section uses the **Standard grid**, resizing one panel resizes all panels in the section.
  - If a section uses the **Custom grid**, you can customize the size of each panel separately.
1. If a section is paginated, you can customize the number of panels to show on a page:
  1. At the top of the section, click **1 to <X> of <Y>**, where `<X>` is the number of visible panels and `<Y>` is the total number of panels.
  1. Choose how many panels to show per page, up to 100.
1. To delete a panel from a section:
  1. Hover over the panel, then click its action `...` menu.
  1. Click **Delete**.
  
If you reset a workspace to an automated workspace, all deleted panels appear again.

### Rename a section

To rename a section, click its action `...` menu, then click **Rename section**.

### Delete a section

To delete a section, click its `...` menu, then click **Delete section**. This removes the section and its panels.
