---
menu:
  default:
    identifier: cascade-settings
    parent: w-b-app-ui-reference
title: Manage workspace, section, and panel settings
url: guides/app/features/cascade-settings
---

Within a given workspace page there are three different setting levels: workspaces, sections, and panels. [Workspace settings]({{< relref "#workspace-settings" >}}) apply to the entire workspace. [Section settings]({{< relref "#section-settings" >}}) apply to all panels within a section. [Panel settings]({{< relref "#panel-settings" >}}) apply to individual panels. 

## Workspace settings

Workspace settings apply to all sections and all panels within those sections. You can edit two types of workspace settings: [Workspace layout]({{< relref "#workspace-layout-options" >}}) and [Line plots]({{< relref "#line-plots-options" >}}). **Workspace layouts** determine the structure of the workspace, while **Line plots** settings control the default settings for line plots in the workspace.ne plots** settings control the default settings for line plots in the workspace.

To edit settings that apply to the overall structure of this workspace:

1. Navigate to your project workspace.
2. Click the gear icon next to the **New report** button to view the workspace settings.
3. Choose **Workspace layout** to change the workspace's layout, or choose **Line plots** to configure default settings for line plots in the workspace.
{{< img src="/images/app_ui/workspace_settings.png" alt="Workspace settings gear icon" >}}

{{% alert %}}
After customizing your workspace, you can use _workspace templates_ to quickly create new workspaces with the same settings. Refer to [Workspace templates]({{< relref "/guides/models/track/workspaces.md#workspace-templates" >}}).
{{% /alert %}}

### Workspace layout options

Configure a workspaces layout to define the overall structure of the workspace. This includes sectioning logic and panel organization. 

{{< img src="/images/app_ui/workspace_layout_settings.png" alt="Workspace layout options" >}}

The workspace layout options page shows whether the workspace generates panels automatically or manually. To adjust a workspace's panel generation mode, refer to [Panels]({{< relref "panels/" >}}).

This table describes each workspace layout option.

| Workspace setting | Description |
| ----- | ----- |
| **Hide empty sections during search** |  Hide sections that do not contain any panels when searching for a panel.|
| **Sort panels alphabetically** | Sort panels in your workspaces alphabetically. |
| **Section organization** | Remove all existing sections and panels and repopulate them with new section names. Groups the newly populated sections either by first or last prefix. |

{{% alert %}}
W&B suggests that you organize sections by grouping the first prefix rather than grouping by the last prefix. Grouping by the first prefix can result in fewer sections and better performance.
{{% /alert %}}

### Line plots options
Set global defaults and custom rules for line plots in a workspace by modifying the **Line plots** workspace settings.

{{< img src="/images/app_ui/workspace_settings_line_plots.png" alt="Line plot settings" >}}

You can edit two main settings within **Line plots** settings: **Data** and **Display preferences**. The **Data** tab contains the following settings:


| Line plot setting | Description |
| ----- | ----- |
| **X axis** |  The scale of the x-axis in line plots. The x-axis is set to **Step** by default. See the proceeding table for the list of x-axis options. |
| **Range** |  Minimum and maximum settings to display for x axis. |
| **Smoothing** | Change the smoothing on the line plot. For more information about smoothing, see [Smooth line plots]({{< relref "/guides/models/app/features/panels/line-plot/smoothing.md" >}}). |
| **Outliers** | Rescale to exclude outliers from the default plot min and max scale. |
| **Point aggregation method** | Improve data visualization accuracy and performance. See [Point aggregation]({{< relref "/guides/models/app/features/panels/line-plot/sampling.md" >}}) for more information. |
| **Max number of runs or groups** | Limit the number of runs or groups displayed on the line plot. |

In addition to **Step**, there are other options for the x-axis:

| X axis option | Description |
| ------------- | ----------- |
| **Relative Time (Wall)**| Timestamp since the process starts. For example, suppose start a run and resume that run the next day. If you then log something, the recorded point is 24 hours.|
| **Relative Time (Process)** | Timestamp inside the running process. For example, suppose you start a run and let it continue for 10 seconds. The next day you resume that run. The point is recorded as 10 seconds. |
| **Wall Time** | Minutes elapsed since the start of the first run on the graph. |
| **Step** | Increments each time you call `wandb.Run.log()`.|



{{% alert %}}
For information on how to edit an individual line plot, see [Edit line panel settings]({{< relref "/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" >}}) in Line plots. 
{{% /alert %}}


Within the **Display preferences** tab, you can toggle the proceeding settings:

| Display preference | Description |
| ----- | ----- |
| **Remove legends from all panels** | Remove the panel's legend |
| **Display colored run names in tooltips** | Show the runs as colored text within the tooltip |
| **Only show highlighted run in companion chart tooltip** | Display only highlighted runs in chart tooltip |
| **Number of runs shown in tooltips** | Display the number of runs in the tooltip |
| **Display full run names on the primary chart tooltip**| Display the full name of the run in the chart tooltip |




## Section settings

Section settings apply to all panels within that section. Within a workspace section you can sort panels, rearrange panels, and rename the section name.

Modify section settings by selecting the three horizontal dots (**...**) in the upper right corner of a section.

{{< img src="/images/app_ui/section_settings.png" alt="Section settings menu" >}}

From the dropdown, you can edit the following settings that apply to the entire section:

| Section setting | Description |
| ----- | ----- |
| **Rename a section** | Rename the name of the section |
| **Sort panels A-Z** | Sort panels within a section alphabetically |
| **Rearrange panels** | Select and drag a panel within a section to manually order your panels |

The proceeding animation demonstrates how to rearrange panels within a section:

{{< img src="/images/app_ui/rearrange_panels.gif" alt="Rearranging panels" >}}

{{% alert %}}
In addition to the settings described in the preceding table, you can also edit how sections appear in your workspaces such as **Add section below**, **Add section above**, **Delete section**, and **Add section to report**. 
{{% /alert %}}

## Panel settings

Customize an individual panel's settings to compare multiple lines on the same plot, calculate custom axes, rename labels, and more. To edit a panel's settings:

1. Hover your mouse over the panel you want to edit. 
2. Select the pencil icon that appears.
{{< img src="/images/app_ui/panel_settings.png" alt="Panel edit icon" >}}
3. Within the modal that appears, you can edit settings related to the panel's data, display preferences, and more.
{{< img src="/images/app_ui/panel_settings_modal.png" alt="Panel settings modal" >}}

For a complete list of settings you can apply to a panel, see [Edit line panel settings]({{< relref "/guides/models/app/features/panels/line-plot/#edit-line-panel-settings" >}}).