---
title: WorkspaceSettings
object_type: python_sdk_workspaces_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `WorkspaceSettings`
Settings for the workspace, typically seen at the top of the workspace in the UI. 

This object includes settings for the x-axis, smoothing, outliers, panels, tooltips, runs, and panel query bar. 

Settings applied here can be overrided by more granular Section and Panel settings in this priority: Workspace < Section < Panel 



**Attributes:**
 
 - `x_axis` (str):  X-axis metric name setting. 
 - `x_min` (Optional[float]):  Minimum value for the x-axis. 
 - `x_max` (Optional[float]):  Maximum value for the x-axis. 
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']):  Smoothing  type applied to all panels. 
 - `smoothing_weight` (int):  Smoothing weight applied to all panels. 
 - `ignore_outliers` (bool):  Ignore outliers in all panels. 
 - `sort_panels_alphabetically` (bool):  Sorts panels in all sections alphabetically. 
 - `group_by_prefix` (Literal["first", "last"]):  Group panels by the first or up to last  prefix (first or last). Default is set to `last`. 
 - `remove_legends_from_panels` (bool):  Remove legends from all panels. 
 - `tooltip_number_of_runs` (Literal["default", "all", "none"]):  The number of runs to show in the tooltip. 
 - `tooltip_color_run_names` (bool):  Whether to color run names in the tooltip to  match the runset (True) or not (False). Default is set to `True`. 
 - `max_runs` (int):  The maximum number of runs to show per panel (this will be the first 10 runs in the runset). 
 - `point_visualization_method` (Literal["line", "point", "line_point"]):  The visualization method for points. 
 - `panel_search_query` (str):  The query for the panel search bar (can be a regex expression). 
 - `auto_expand_panel_search_results` (bool):  Whether to auto expand the panel search results. 




