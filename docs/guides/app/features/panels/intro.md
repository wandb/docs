---
slug: /guides/app/features/panels
displayed_sidebar: default
title: Panels
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Use panel visualizations to explore your logged data, the relationships between hyperparameters and output metrics, and more. 

## Check workspace settings

A workspace's type indicates whether the workspace generates panels automatically. Next to the name of your workspace is a clipboard icon.

- If the icon is a green diamond, automatic panel generation is enabled.
- If the icon is a purple safety pin and ruler, automatic panel generation is disabled.

To configure panel generation, [reset the workspace](#reset-a-workspace).

## Reset a workspace

By default, a workspace automatically generates panels for all keys [`log`](../../../../ref/python/log.md) in the project. This can help you get started quickly by visualizing all available data for the project.

:::info
When a workspace has automatic panel generation enabled, W&B uses the key value you specify with [`log`](../../../../ref/python/log.md) to determine whether or not to create a new panel. 
:::


If desired, you can configure the workspace to display only those panels you add manually. Resetting a workspace removes all custom panels.

To change the type for a workspace:

1. At the top of the workspace, click the `...` menu, then click **Reset workspace**.
2. To generate panels automatically, select **Automatic**, then click **Generate automated workspace**.
3. To generate panels manually instead, select **Manual**, then click **Get started**.


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

## Duplicate a panel

To duplicate a panel:

1. At the top of the panel, click the `...` menu.
2. Click **Duplicate**.
3. If desired, customize the duplicate panel.

## Add multiple panels
You can add up to 500 panels at a time. To add multiple panels:

1. Within your project workspace, choose the **Add panels** button
2. Choose **Quick add**
2. Provide a regular expression within the search field
3. Select the **Add all** button
![](/images/app_ui/bulk_panels.gif)

:::note
The **Add all** button appears only if a regular expression match occurs.
:::

## Remove a panel

1. Hover your mouse in the upper corner of the panel you want to remove
2. Select the three horizontal dots (**...**) that appear
3. From the dropdown, select **Delete**

## Remove all panels

To remove all customization from an automatic workspace, [reset the workspace](#reset-a-workspace).

To remove all panels from a manual workspace and leave it empty:

1. Within your project workspace, select the three horizontal dots (**...**) next to the panel search bar
2. Select **Clear all panels**

<!-- ## Add a section -->

<!-- Delete a section -->

