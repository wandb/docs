---
title: Workspaces
---
{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/workspaces/interface.py" >}}

<!-- markdownlint-turnedoff -->

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}


# <kbd>module</kbd> `wandb_workspaces.workspaces`
Python library for programmatically working with W&B Workspace API. 

```python
# How to import
import wandb_workspaces.workspaces as ws

# Example of creating a workspace
ws.Workspace(
     name="Example W&B Workspace",
     entity="entity", # entity that owns the workspace
     project="project", # project that the workspace is associated with
     sections=[
         ws.Section(
             name="Validation Metrics",
             panels=[
                 wr.LinePlot(x="Step", y=["val_loss"]),
                 wr.BarPlot(metrics=["val_accuracy"]),
                 wr.ScalarChart(metric="f1_score", groupby_aggfunc="mean"),
             ],
             is_open=True,
         ),
     ],
)
workspace.save()
```

---



## <kbd>class</kbd> `RunSettings`
Settings for a run in a runset (left hand bar). 



**Attributes:**
 
 - `color` (str): The color of the run in the UI. Can be hex (#ff0000), css color (red), or rgb (rgb(255, 0, 0)) 
 - `disabled` (bool): Whether the run is deactivated (eye closed in the UI). Default is set to `False`. 







---



## <kbd>class</kbd> `RunsetSettings`
Settings for the runset (the left bar containing runs) in a workspace. 



**Attributes:**
 
 - `query` (str): A query to filter the runset (can be a regex expr, see next param). 
 - `regex_query` (bool): Controls whether the query (above) is a regex expr. Default is set to `False`. 
 - `filters` `(LList[expr.FilterExpr])`: A list of filters to apply to the runset. Filters are AND'd together. See FilterExpr for more information on creating filters. 
 - `groupby` `(LList[expr.MetricType])`: A list of metrics to group by in the runset. Set to `Metric`, `Summary`, `Config`, `Tags`, or `KeysInfo`. 
 - `order` `(LList[expr.Ordering])`: A list of metrics and ordering to apply to the runset. 
 - `run_settings` `(Dict[str, RunSettings])`: A dictionary of run settings, where the key is the run's ID and the value is a RunSettings object. 







---



## <kbd>class</kbd> `Section`
Represents a section in a workspace. 



**Attributes:**
 
 - `name` (str): The name/title of the section. 
 - `panels` `(LList[PanelTypes])`: An ordered list of panels in the section. By default, first is top-left and last is bottom-right. 
 - `is_open` (bool): Whether the section is open or closed. Default is closed. 
 - `layout_settings` `(Literal[`standard`, `custom`])`: Settings for panel layout in the section. 
 - `panel_settings`: Panel-level settings applied to all panels in the section, similar to `WorkspaceSettings` for a `Section`. 







---



## <kbd>class</kbd> `SectionLayoutSettings`
Panel layout settings for a section, typically seen at the top right of the section of the W&B App Workspace UI. 



**Attributes:**
 
 - `layout` `(Literal[`standard`, `custom`])`: The layout of panels in the section. `standard` follows the default grid layout, `custom` allows per per-panel layouts controlled by the individual panel settings. 
 - `columns` (int): In a standard layout, the number of columns in the layout. Default is 3. 
 - `rows` (int): In a standard layout, the number of rows in the layout. Default is 2. 







---



## <kbd>class</kbd> `SectionPanelSettings`
Panel settings for a section, similar to `WorkspaceSettings` for a section. 

Settings applied here can be overrided by more granular Panel settings in this priority: Section < Panel. 



**Attributes:**
 
 - `x_axis` (str): X-axis metric name setting. By default, set to `Step`. 
 - `x_min Optional[float]`: Minimum value for the x-axis. 
 - `x_max Optional[float]`: Maximum value for the x-axis. 
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']): Smoothing type applied to all panels. 
 - `smoothing_weight` (int): Smoothing weight applied to all panels. 







---



## <kbd>class</kbd> `Workspace`
Represents a W&B workspace, including sections, settings, and config for run sets. 



**Attributes:**
 
 - `entity` (str): The entity this workspace will be saved to (usually user or team name). 
 - `project` (str): The project this workspace will be saved to. 
 - `name`: The name of the workspace. 
 - `sections` `(LList[Section])`: An ordered list of sections in the workspace. The first section is at the top of the workspace. 
 - `settings` `(WorkspaceSettings)`: Settings for the workspace, typically seen at the top of the workspace in the UI. 
 - `runset_settings` `(RunsetSettings)`: Settings for the runset (the left bar containing runs) in a workspace. 


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
 
 - `Workspace`: The updated workspace with the saved internal name and ID. 

---



### <kbd>method</kbd> `save_as_new_view`

```python
save_as_new_view()
```

Save the current workspace as a new view to W&B. 



**Returns:**
 
 - `Workspace`: The updated workspace with the saved internal name and ID.

---



## <kbd>class</kbd> `WorkspaceSettings`
Settings for the workspace, typically seen at the top of the workspace in the UI. 

This object includes settings for the x-axis, smoothing, outliers, panels, tooltips, runs, and panel query bar. 

Settings applied here can be overrided by more granular Section and Panel settings in this priority: Workspace < Section < Panel 



**Attributes:**
 
 - `x_axis` (str): X-axis metric name setting. 
 - `x_min` `(Optional[float])`: Minimum value for the x-axis. 
 - `x_max` `(Optional[float])`: Maximum value for the x-axis. 
 - `smoothing_type` `(Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none'])`: Smoothing type applied to all panels. 
 - `smoothing_weight` (int): Smoothing weight applied to all panels. 
 - `ignore_outliers` (bool): Ignore outliers in all panels. 
 - `sort_panels_alphabetically` (bool): Sorts panels in all sections alphabetically. 
 - `group_by_prefix` `(Literal[`first`, `last`])`: Group panels by the first or up to last prefix (first or last). Default is set to `last`. 
 - `remove_legends_from_panels` (bool): Remove legends from all panels. 
 - `tooltip_number_of_runs` `(Literal[`default`, `all`, `none`])`: The number of runs to show in the tooltip. 
 - `tooltip_color_run_names` (bool): Whether to color run names in the tooltip to match the runset (True) or not (False). Default is set to `True`. 
 - `max_runs` (int): The maximum number of runs to show per panel (this will be the first 10 runs in the runset). 
 - `point_visualization_method` `(Literal[`line`, `point`, `line_point`])`: The visualization method for points. 
 - `panel_search_query` (str): The query for the panel search bar (can be a regex expression). 
 - `auto_expand_panel_search_results` (bool): Whether to auto expand the panel search results. 





