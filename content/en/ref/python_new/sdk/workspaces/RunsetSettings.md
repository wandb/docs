---
title: RunsetSettings
object_type: python_sdk_workspaces_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `RunsetSettings`
Settings for the runset (the left bar containing runs) in a workspace. 



**Attributes:**
 
 - `query` (str):  A query to filter the runset (can be a regex expr, see next param). 
 - `regex_query` (bool):  Controls whether the query (above) is a regex expr. Default is set to `False`. 
 - `filters` (LList[expr.FilterExpr]):  A list of filters to apply to the runset.  Filters are AND'd together. See FilterExpr for more information on creating filters. 
 - `groupby` (LList[expr.MetricType]):  A list of metrics to group by in the runset. Set to  `Metric`, `Summary`, `Config`, `Tags`, or `KeysInfo`. 
 - `order` (LList[expr.Ordering]):  A list of metrics and ordering to apply to the runset. 
 - `run_settings` (Dict[str, RunSettings]):  A dictionary of run settings, where the key  is the run's ID and the value is a RunSettings object. 




