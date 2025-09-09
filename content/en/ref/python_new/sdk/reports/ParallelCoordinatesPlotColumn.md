---
title: ParallelCoordinatesPlotColumn
object_type: python_sdk_reports_namespace
data_type_classification: class
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/ >}}




## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
A column within a parallel coordinates plot.  The order of `metric`s specified determine the order of the parallel axis (x-axis) in the parallel coordinates plot. 



**Attributes:**
 
 - `metric` (str | Config | SummaryMetric):  The name of the  metric logged to your W&B project that the report pulls information from. 
 - `display_name` (Optional[str]):  The name of the metric 
 - `inverted` (Optional[bool]):  Whether to invert the metric. 
 - `log` (Optional[bool]):  Whether to apply a log transformation to the metric. 




