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

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use panel visualizations to explore your logged data, the relationships between hyperparameters and output metrics, and more. 

## Add a single panel

1. Within your workspace, navigate to the section you want to add a panel to
2. Choose the Add panel button
3. From the dropdown, select a type of panel to add
{{< img src="/images/app_ui/add_single_panel.gif" alt="" >}} 
4. (Optional) If prompted, define parameters for the plot. 

<Tabs
  defaultValue="quick"
  values={[
    {label: 'Add a plot from a logged value', value: 'quick'},
    {label: 'Add a custom plot', value: 'single'},
  ]}>
  <TabItem value="quick">

1. Within your project workspace, choose the **Add panels** button
2. Select **Quick add**
2. Provide a regular expression within the search field or select a key from the **KEYS** dropdown.

  </TabItem>
  <TabItem value="single">

1. Within your project workspace, choose the **Add panels** button
2. Select the type of chart you want to add from the **CHARTS** dropdown
3. Based on the chart type, provide the necessary parameters
3. Select **Apply**

  </TabItem> 
</Tabs>


{{% alert title="Undo changes to your workspace" %}}
Select the undo button (arrow that points left) to undo any unwanted changes.
{{% /alert %}}


## Add multiple panels
Add multiple panels to your workspace at the same time. You can add up to 500 panels at a time.

1. Within your project workspace, choose the **Add panels** button
2. Choose **Quick add**
2. Provide a regular expression within the search field
3. Select the **Add all** button
{{< img src="/images/app_ui/bulk_panels.gif" alt="" >}}

{{% alert %}}
The **Add all** button appears only if a regular expression match occurs.
{{% /alert %}}



## Activate or deactivate auto generated panels

By default, W&B generates a panel for each unique metric you [`log`](../../../../ref/python/log.md) with the Python SDK. 

{{% alert %}}
W&B uses the key value you specify with [`log`](../../../../ref/python/log.md) to determine whether or not to create a new panel. 
{{% /alert %}}

To activate or deactivate auto generated panels:

1. Navigate to your project's workspace
2. Select on the gear icon in the upper right hand corner
3. A modal appears, choose **Sections**
4. Toggle the **Panel generation** option to desired state
{{< img src="/images/app_ui/panel_generation.png" alt="" >}}

### Check auto generated panel settings
Each workspace indicates whether or not the workspace automatically generates panels. Next to the name of your workspace is a clipboard icon. If the icon is red, panels are not automatically generated. If the panel is green, panels are automatically created each time you log a unique metric.

Example of workspace with panel auto generation off:
{{< img src="/images/app_ui/auto_panel_off.png" alt="" >}}

Example of workspace with panel auto generation on:
{{< img src="/images/app_ui/auto_panel_on.png" alt="" >}}


## Remove a panel

1. Hover your mouse in the upper corner of the panel you want to remove
2. Select the three horizontal dots (**...**) that appear
3. From the dropdown, select **Delete**

## Remove all panels

1. Within your project workspace, select the three horizontal dots (**...**) next to the panel search bar
2. Select **Clear all panels**

{{% alert %}}
Clearing panels in your workspace deactivates automatic panel generation.
{{% /alert %}}

<!-- ## Add a section -->

<!-- Delete a section -->