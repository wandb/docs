---
slug: /guides/app/features/panels
displayed_sidebar: default
title: Panels
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use panel visualizations to explore your logged data, the relationships between hyperparameters and output metrics, and more. 

## Add a single panel

1. Within your workspace, navigate to the section you want to add a panel to
2. Choose the Add panel button
3. From the dropdown, select a type of panel to add
![](/images/app_ui/add_single_panel.gif) 
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


:::tip Undo changes to your workspace
Select the undo button (arrow that points left) to undo any unwanted changes.
:::

## Add multiple panels
Add multiple panels to your workspace at the same time. You can add up to 500 panels at a time.

1. Within your project workspace, choose the **Add panels** button
2. Choose **Quick add**
2. Provide a regular expression within the search field
3. Select the **Add all** button
![](/images/app_ui/bulk_panels.gif)

:::note
The **Add all** button appears only if a regular expression match occurs.
:::



## Activate or deactivate auto generated panels

By default, W&B generates a panel for each unique metric you [`log`](../../../../ref/python/log.md) with the Python SDK. 

:::info
W&B uses the key value you specify with [`log`](../../../../ref/python/log.md) to determine whether or not to create a new panel. 
:::

To activate or deactivate auto generated panels:

1. Navigate to your project's workspace
2. Select on the gear icon in the upper right hand corner
3. A modal appears, choose **Sections**
4. Toggle the **Panel generation** option to desired state
![](/images/app_ui/panel_generation.png)

### Check auto generated panel settings
Each workspace indicates whether or not the workspace automatically generates panels. Next to the name of your workspace is a clipboard icon. If the icon is red, panels are not automatically generated. If the panel is green, panels are automatically created each time you log a unique metric.

Example of workspace with panel auto generation off:
![](/images/app_ui/auto_panel_off.png)

Example of workspace with panel auto generation on:
![](/images/app_ui/auto_panel_on.png)


## Remove a panel

1. Hover your mouse in the upper corner of the panel you want to remove
2. Select the three horizontal dots (**...**) that appear
3. From the dropdown, select **Delete**

## Remove all panels

1. Within your project workspace, select the three horizontal dots (**...**) next to the panel search bar
2. Select **Clear all panels**

:::note
Clearing panels in your workspace deactivates automatic panel generation.
:::

<!-- ## Add a section -->

<!-- Delete a section -->

