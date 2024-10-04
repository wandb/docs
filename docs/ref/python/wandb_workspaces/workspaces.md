import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx'

# Workspaces

<CTAButtons githubLink='https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py'/>

<!-- markdownlint-disable -->



# <kbd>module</kbd> `wandb_workspaces.workspaces`
Python library for programmatically working with W&B Workspace API. 

```python
# How to import
import wandb_workspaces.workspaces
```

---



## <kbd>class</kbd> `RunSettings`
Settings for a run in a runset (left hand bar). 



**Attributes:**
 
 - `color`:  The color of the run in the UI.  Can be hex (#ff0000), css color (red), or rgb (rgb(255, 0, 0)) 
 - `disabled`:  Whether the run is disabled (eye closed in the UI). 







---



## <kbd>class</kbd> `RunsetSettings`
Settings for the runset (the left bar containing runs) in a workspace. 



**Attributes:**
 
 - `query`:  A query to filter the runset (can be a regex expr, see next param). 
 - `regex_query`:  Controls whether the query (above) is a regex expr. 
 - `filters`:  A list of filters to apply to the runset.  Filters are AND'd together. See FilterExpr for more information on creating filters. 
 - `groupby`:  A list of metrics to group by in the runset. 
 - `order`:  A list of metrics and ordering to apply to the runset. 
 - `run_settings`:  A dictionary of run settings, where the key is the run's ID and the value is a RunSettings object. 







---



## <kbd>class</kbd> `Section`
Represents a section in a workspace. 



**Attributes:**
 
 - `name`:  The name/title of the section. 
 - `panels`:  An ordered list of panels in the section.  By default, first is top-left and last is bottom-right. 
 - `is_open`:  Whether the section is open or closed.  Default is closed. 
 - `layout_settings`:  Settings for panel layout in the section. 
 - `panel_settings`:  Panel-level settings applied to all panels in the section, similar to WorkspaceSettings for this Section. 







---



## <kbd>class</kbd> `SectionLayoutSettings`
Panel layout settings for a section, typically seen at the top right of the section of the W&B App Workspace UI. 



**Attributes:**
 
 - `layout`:  In a standard layout, the number of columns in the layout. 
 - `columns`:  In a standard layout, the number of columns in the layout. 
 - `rows`:  In a standard layout, the number of rows in the layout. 







---



## <kbd>class</kbd> `SectionPanelSettings`
Panel settings for a section, similar to `WorkspaceSettings` for a section. 

Settings applied here can be overrided by more granular Panel settings in this priority: Section < Panel. 



**Attributes:**
 
 - `x_axis`:  X-axis metric name setting. 
 - `x_min`:  Minimum value for the x-axis. 
 - `x_max`:  Maximum value for the x-axis. 
 - `smoothing_type`:  Smoothing type applied to all panels. 
 - `smoothing_weight`:  Smoothing weight applied to all panels. 







---



## <kbd>class</kbd> `Workspace`
Represents a W&B workspace, including sections, settings, and config for run sets. 



**Attributes:**
 
 - `entity`:  The entity this workspace will be saved to (usually user or team name). 
 - `project`:  The project this workspace will be saved to. 
 - `name`:  The name of the workspace. 
 - `sections`:  An ordered list of sections in the workspace.  The first section is at the top of the workspace. 
 - `settings`:  Settings for the workspace, typically seen at the top of the workspace in the UI. 
 - `runset_settings`:  Settings for the runset (the left bar containing runs) in a workspace. 


---

#### <kbd>property</kbd> url

The URL to the workspace in the W&B app. 



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str)
```

Get a workspace from a URL. 

---



### <kbd>method</kbd> `save`

```python
save()
```

Save the current workspace to W&B. 



**Returns:**
 
 - `Workspace`:  The updated workspace with the saved internal name and ID. 

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

Save the current workspace as a new view to W&B. 



**Returns:**
 
 - `Workspace`:  The updated workspace with the saved internal name and ID.

---



## <kbd>class</kbd> `WorkspaceSettings`
Settings for the workspace, typically seen at the top of the workspace in the UI. 

This object includes settings for the x-axis, smoothing, outliers, panels, tooltips, runs, and panel query bar. 

Settings applied here can be overrided by more granular Section and Panel settings in this priority: Workspace < Section < Panel 



**Attributes:**
 
 - `x_axis`:  X-axis metric name setting. 
 - `x_min`:  Minimum value for the x-axis. 
 - `x_max`:  Maximum value for the x-axis. 
 - `smoothing_type`:  Smoothing type applied to all panels. 
 - `smoothing_weight`:  Smoothing weight applied to all panels. 
 - `ignore_outliers`:  Ignore outliers in all panels. 
 - `sort_panels_alphabetically`:  Sorts panels in all sections alphabetically. 
 - `group_by_prefix`:  Group panels by the first or up to last prefix (first or last). 





