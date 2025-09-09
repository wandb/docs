---
title: Workspace
object_type: python_sdk_workspaces_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `Workspace`
Represents a W&B workspace, including sections, settings, and config for run sets. 



**Attributes:**
 
 - `entity` (str):  The entity this workspace will be saved to (usually user or team name). 
 - `project` (str):  The project this workspace will be saved to. 
 - `name`:  The name of the workspace. 
 - `sections` (LList[Section]):  An ordered list of sections in the workspace.  The first section is at the top of the workspace. 
 - `settings` (WorkspaceSettings):  Settings for the workspace, typically seen at  the top of the workspace in the UI. 
 - `runset_settings` (RunsetSettings):  Settings for the runset  (the left bar containing runs) in a workspace. 
 - `auto_generate_panels` (bool):  Whether to automatically generate panels for all keys logged in this project.  Recommended if you would like all available data to be visualized by default.  This can only be set during workspace creation and cannot be modified afterward. 


---

### <kbd>property</kbd> Workspace.auto_generate_panels





---

### <kbd>property</kbd> Workspace.url

The URL to the workspace in the W&B app. 



---

### <kbd>classmethod</kbd> `Workspace.from_url`

```python
from_url(url: str)
```

Get a workspace from a URL. 

---

### <kbd>method</kbd> `Workspace.save`

```python
save()
```

Save the current workspace to W&B. 



**Returns:**
 
 - `Workspace`:  The updated workspace with the saved internal name and ID. 

---

### <kbd>method</kbd> `Workspace.save_as_new_view`

```python
save_as_new_view()
```

Save the current workspace as a new view to W&B. 



**Returns:**
 
 - `Workspace`:  The updated workspace with the saved internal name and ID. 

