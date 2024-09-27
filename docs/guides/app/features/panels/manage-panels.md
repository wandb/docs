---
display_sidebar: default
---

# Manage and organize panels

Add, remove, and organize panels within your project's workspace.

## Activate or deactivate auto generated panels

By default, W&B generates a panel for each unique metric you [`log`](../../../../ref/python/log.md) with the Python SDK. 

:::info
W&B uses the key value you specify with [`log`](../../../../ref/python/log.md) to determine whether or not to create a new panel. 
:::

To activate or deactivate this functionality: 

1. Navigate to your project's workspace
2. Select on the gear icon in the upper right hand corner
3. A modal appears, choose **Sections**
4. Toggle the **Panel generation** option to desired state
![](/images/app_ui/panel_generation.png)

### Check auto generated panel settings
Each workspace indicates whether or not the workspace automatically generates panels. Next to the name of your workspace you will see a clipboard icon. If the icon is red, panels are not automatically generated. If the the panel is green, panels are automatically created each time you log a unique metric.

Example of workspace with panel auto generation off:
![](/images/app_ui/auto_panel_off.png)

Example of workspace with panel auto generation on:
![](/images/app_ui/auto_panel_on.png)

## Clear all panels

1. Within your project workspace, select the three horizontal dots (**...**) next to the panel search bar
2. Select **Clear all panels**

:::note
Clearing panels in your workspace deactivates automatic panel generation.
:::

:::tip Undo changes to your workspace
Select the undo button (arrow that points left) to undo any changes you make to your workspace
:::



## Add multiple panels

1. Within your project workspace, choose the **Add panels** button
2. Provide a regex expression within the search field
3. Select the **Add all** button

![](/images/app_ui/bulk_panels.gif)

:::note
The **Add all** appears only if a regex match occurs.
:::


## Organize workspace