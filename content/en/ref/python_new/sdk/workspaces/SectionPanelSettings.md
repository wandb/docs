---
title: SectionPanelSettings
object_type: python_sdk_workspaces_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `SectionPanelSettings`
Panel settings for a section, similar to `WorkspaceSettings` for a section. 

Settings applied here can be overrided by more granular Panel settings in this priority: Section < Panel. 



**Attributes:**
 
 - `x_axis` (str):  X-axis metric name setting. By default, set to "Step". 
 - `x_min Optional[float]`:  Minimum value for the x-axis. 
 - `x_max Optional[float]`:  Maximum value for the x-axis. 
 - `smoothing_type` (Literal['exponentialTimeWeighted', 'exponential', 'gaussian', 'average', 'none']):  Smoothing  type applied to all panels. 
 - `smoothing_weight` (int):  Smoothing weight applied to all panels. 




