---
title: Section
object_type: python_sdk_workspaces_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `Section`
Represents a section in a workspace. 



**Attributes:**
 
 - `name` (str):  The name/title of the section. 
 - `panels` (LList[PanelTypes]):  An ordered list of panels in the section. By default, first is top-left and last is bottom-right. 
 - `is_open` (bool):  Whether the section is open or closed. Default is closed. 
 - `layout_settings` (Literal["standard", "custom"]):  Settings for panel layout in the section. 
 - `panel_settings`:  Panel-level settings applied to all panels in the section, similar to `WorkspaceSettings` for a `Section`. 




