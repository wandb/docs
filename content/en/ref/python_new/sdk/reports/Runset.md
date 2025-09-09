---
title: Runset
object_type: python_sdk_reports_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `Runset`
A set of runs to display in a panel grid. 



**Attributes:**
 
 - `entity` (str):  An entity that owns or has the correct  permissions to the project where the runs are stored. 
 - `project` (str):  The name of the project were the runs are stored. 
 - `name` (str):  The name of the run set. Set to `Run set` by default. 
 - `query` (str):  A query string to filter runs. 
 - `filters` (Optional[str]):  A filter string to filter runs. 
 - `groupby` (LList[str]):  A list of metric names to group by. Supported formats are: 
        - "group" or "run.group" to group by a run attribute 
        - "config.param" to group by a config parameter 
        - "summary.metric" to group by a summary metric 
 - `order` (LList[OrderBy]):  A list of `OrderBy` objects to order by. 
 - `custom_run_colors` (LList[OrderBy]):  A dictionary mapping run IDs to colors. 




