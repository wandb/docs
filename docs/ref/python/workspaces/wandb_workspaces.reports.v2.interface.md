# Reports

<!-- markdownlint-disable -->

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2.py#L0"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
Python library for programmatically working with Weights & Biases Reports API. 

```python
# How to import
import wandb_workspaces.reports.v2
```

---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1758"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `BarPlot`
A panel object that shows a 2D bar plot. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. metrics LList[MetricType]:  
 - `orientation Literal["v", "h"]`:  The orientation of the bar plot.  Set to either vertical ("v") or horizontal ("h"). Defaults to horizontal ("h"). 
 - `range_x` (tuple):  Tuple that specifies the range of the x-axis.  
 - `title_x` (Optional[str]):  The label of the x-axis. 
 - `title_y` (Optional[str]):  The label of the y-axis. 
 - `groupby` (Optional[str]):  Group runs based on a metric logged to your W&B project that the  report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]):  Aggregate runs with specified  function. Options include "mean", "min", "max", "median", "sum", "samples", or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]):   Group runs based on a range. Options  include "minmax", "stddev", "stderr", "none", "samples", or `None`. 
 - `max_runs_to_show` (Optional[int]):  The maximum number of runs to show on the plot. 
 - `max_bars_to_show` (Optional[int]):  The maximum number of bars to show on the bar plot. 
 - `custom_expressions` (Optional[LList[str]]):  A list of custom expressions to be used in the bar plot. 
 - `legend_template` (Optional[str]):  The template for the legend. 
 - `font_size` ( Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `line_titles` (Optional[dict]):  The titles of the lines. The keys are the line names and the values are the titles. 
 - `line_colors` (Optional[dict]):  The colors of the lines. The keys are the line names and the values are the colors. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L169"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Block`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L482"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `BlockQuote`
A block of quoted text. 



**Attributes:**
 
 - `text`:  The text of the block quote. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L588"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CalloutBlock`
A block of callout text. 



**Attributes:**
 
 - `text`:  The callout text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L440"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedList`
A list of items with checkboxes. Add one or more `CheckedListItem` within `CheckedList`. 



**Attributes:**
 
 - `items`:  A list of one or more `CheckedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L365"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedListItem`
A list item with a checkbox. Add one ore more `CheckedListItem` within `CheckList`. 



**Attributes:**
 
 - `text`:  The text of the list item. 
 - `checked`:  Whether the checkbox is checked. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L499"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeBlock`
A block of code. 



**Attributes:**
 
 - `code`:  The code in the block. 
 - `language`:  The language of the code. The language specified  is used for syntax highlighting. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1913"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeComparer`
A panel object that compares the code between two different runs. 



**Attributes:**
 
 - `diff` (Required):  How to display code differences.  Options include "split" and "unified".  







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L126"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Config`
A configuration for a metric. 



**Attributes:**
 
 - `name`:  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2157"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CustomChart`
A panel that shows a custom chart. The chart is defined by a query. 



**Attributes:**
 
 - `query` (dict):  The query that defines the custom chart. The key is the name of the field, and the value is the query. 
 - `chart_name` (str):  The title of the custom chart. 
 - `chart_fields` (dict):  Key-value pairs that define the axis of the  plot. Where the key is the label, and the value is the metric. 
 - `chart_strings` (dict):  Key-value pairs that define the strings in the chart. 






---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L685"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Gallery`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L672"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryReport`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L677"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryURL`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1492"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GradientPoint`
A point in a gradient. 



**Attributes:**
 
 - `color`:  The color of the point. 
 - `offset`:  The position of the point in the gradient. The value should be between 0 and 100. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L225"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H1`
Creates an H1 HTML tag with the text specified. 



**Attributes:**
 
 - `text`:  The text of the heading. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L247"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H2`
Creates an H2 HTML tag with the text specified. 



**Attributes:**
 
 - `text`:  The text of the heading. 
 - `collapsed_blocks`:  The blocks to show when the heading is collapsed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L270"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H3`
Creates an H3 HTML tag with the text specified. 



**Attributes:**
 
 - `text`:  The text of the heading. 
 - `collapsed_blocks`:  The blocks to show when the heading is collapsed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L206"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Heading`
A heading block. INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L610"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `HorizontalRule`
HTML horizontal line. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L563"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Image`
A block that renders an image. 



**Attributes:**
 
 - `url`:  The URL of the image. 
 - `caption`:  The caption of the image. Caption appears underneath the image. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L321"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineCode`
A block of inline code. Does not add newline character after code. This differs from `CodeBlock` which is an HTML block with code. 



**Attributes:**
 
 - `text`:  The code. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L309"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineLatex`
A block of inline LaTeX. Does not add newline character after LaTeX. This differs from `LatexBlock` which is an HTML block with LaTeX. 



**Attributes:**
 
 - `text`:  The LaTeX code. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L546"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LatexBlock`
A block of LaTeX text. 



**Attributes:**
 
 - `text`:  The LaTeX text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L146"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Layout`
The layout of a block in a report. Adjusts the size and position of the block. 



**Attributes:**
 
 - `x`:  The x position of the block. 
 - `y`:  The y position of the block. 
 - `w`:  The width of the block. 
 - `h`:  The height of the block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1512"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LinePlot`
A panel object with 2D line plots. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `x` (Optional[MetricType]):  The name of a metric logged to your W&B project that the  report pulls information from. The metric specified is used for the x-axis. 
 - `y` (LList[MetricType]):  One or more metrics logged to your W&B project that the report pulls  information from. The metric specified is used for the y-axis. 
 - `range_x` (tuple):  Tuple that specifies the range of the x-axis.  
 - `range_y` (tuple):  Tuple that specifies the range of the y-axis.  
 - `log_x` (Optional[bool]):  Plots the x-coordinates using a base-10 logarithmic scale. 
 - `log_y` (Optional[bool]):  Plots the y-coordinates using a base-10 logarithmic scale. 
 - `title_x` (Optional[str]):  The label of the x-axis. 
 - `title_y` (Optional[str]):  The label of the y-axis. 
 - `ignore_outliers` (Optional[bool]):  If set to `True`, do not plot outliers. 
 - `groupby` (Optional[str]):  Group runs based on a metric logged to your W&B project that the  report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]):  Aggregate runs with specified  function. Options include "mean", "min", "max", "median", "sum", "samples", or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]):   Group runs based on a range. Options  include "minmax", "stddev", "stderr", "none", "samples", or `None`. 
 - `smoothing_factor` (Optional[float]):  The smoothing factor to apply to the  smoothing type. Accepted values range between 0 and 1. 
 - `smoothing_type Optional[SmoothingType]`:  Apply a filter based on the specified  distribution. Options include "exponentialTimeWeighted", "exponential",  "gaussian", "average", or "none". 
 - `smoothing_show_original` (Optional[bool]):    If set to `True`, show the original data. 
 - `max_runs_to_show` (Optional[int]):  The maximum number of runs to show on the line plot. 
 - `custom_expressions` (Optional[LList[str]]):  Custom expressions to apply to the data. 
 - `plot_type Optional[LinePlotStyle]`:  The type of line plot to generate.  Options include "line", "stacked-area", or "pct-area". 
 - `font_size Optional[FontSize]`:  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `legend_position Optional[LegendPosition]`:  Where to place the legend.  Options include "north", "south", "east", "west", or `None`. 
 - `legend_template` (Optional[str]):  The template for the legend. 
 - `aggregate` (Optional[bool]):  If set to `True`, aggregate the data. 
 - `xaxis_expression` (Optional[str]):  The expression for the x-axis. 
 - `legend_fields` (Optional[LList[str]]):  The fields to include in the legend. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L293"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Link`
A link to a URL. 



**Attributes:**
 
 - `text`:  The text of the link. 
 - `url`:  The URL the link points to. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L420"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `List`
A list of items. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L352"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ListItem`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L528"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownBlock`
A block of markdown text. Useful if you want to write text that uses common markdown syntax. 



**Attributes:**
 
 - `text`:  The markdown text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2129"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownPanel`
A panel that renders a markdown. 



**Attributes:**
 
 - `markdown` (str):  The text you want to appear in the markdown panel. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2095"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MediaBrowser`
A panel that displays media files in a grid layout. 



**Attributes:**
 
 - `num_columns` (Optional[int]):  The number of columns in the grid. 
 - `media_keys` (LList[str]):  A list of media keys that correspond to the media files. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L116"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Metric`
A metric to display in a report. 



**Attributes:**
 
 - `name`:  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L726"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderBy`
A metric to order by. 



**Attributes:**
 
 - `name`:  The name of the metric. 
 - `ascending`:  Whether to sort in ascending order. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L454"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedList`
A list of items in a numbered list. 



**Attributes:**
 
 - `items`:  A list of one or more `OrderedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L385"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedListItem`
A list item in an ordered list. 



**Attributes:**
 
 - `text`:  The text of the list item. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L333"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `P`
A block that contains a paragraph of text. 



**Attributes:**
 
 - `text`:  The text of the paragraph. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L823"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Panel`
A panel to display in a panel grid. 



**Attributes:**
 
 - `layout`:  A `Layout` object. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L837"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `PanelGrid`
An HTML block where you can add `Runset` and `Panel` objects to your project. 

Available panels include: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`. 





**Attributes:**
 
 - `runsets` (list):  A list of one or more `Runset` objects. 
 - `panels` (list):  A list of one or more `Panel` objects. 
 - `active_runset` (int):   The number of runs you want to display within a runset. 
 - `custom_run_colors` (dict):  Key-value pairs where the key is the name of a  run and the value is a color specified by a hexadecimal value. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1977"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlot`
A panel object that shows a parallel coordinates plot. 



**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]):  A list of one  or more `ParallelCoordinatesPlotColumn` objects.  
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `gradient` (Optional[LList[GradientPoint]]):  INSERT 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1941"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
A column within a parallel coordinates plot.  The order of `metric`s specified  determine the order of the parallel axis (x-axis) in the parallel coordinates plot. 



**Attributes:**
 
 - `metric`:  The name of the metric logged to your W&B project that the report pulls information from. 
 - `display_name` (str):  The name of the metric  
 - `inverted` (bool):  Whether to invert the metric. 
 - `log` (bool):  Whether to apply a log transformation to the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2031"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParameterImportancePlot`
A panel that shows how important each hyperparameter is in predicting the chosen metric. 



**Attributes:**
 
 - `with_respect_to` (str):  The metric you want to compare the  parameter importance against. Common metrics might include the loss, accuracy,  and so forth. The metric you specify must be logged within the project  that the report pulls information from. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2707"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Report`
A Data Class that represents a W&B Report. Use the returned object's `blocks` attribute to customize your report. Report Data Class objects do not automatically save. Use the `save()` method to persists changes. 



**Attributes:**
 
 - `project`:  The name of the W&B project you want to load in. The project specified appears in the report's URL. 
 - `entity`:  The W&B entity that owns the report. The entity appears in the report's URL. 
 - `title`:  The title of the report. The title appears at the top of the report as an H1 heading. 
 - `description`:  A description of the report. The description appears underneath the report's title. 
 - `blocks`:  A list of one or more HTML tags, plots, grids, runsets, or more. 
 - `width`:  The width of the report. Options include 'readable', 'fixed', 'fluid'. 




---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2065"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunComparer`
A panel that compares metrics across different runs from the project the report pulls information from. 



**Attributes:**
 
 - `diff_only` (Optional[Literal["split", True]]):  Display only the  difference across runs in a project. You can toggle this feature on and off in the W&B Report UI. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L751"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Runset`
A set of runs to display in a panel grid. 



**Attributes:**
 
 - `entity`:  The entity name. 
 - `project`:  The project name. 
 - `name`:  The name of the run set. 
 - `query`:  A query string to filter runs. 
 - `filters`:  A filter string to filter runs. 
 - `groupby`:  A list of metric names to group by. 
 - `order`:  A list of `OrderBy` objects to order by. 
 - `custom_run_colors`:  A dictionary mapping run IDs to colors. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L103"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroup`
UI element that shows a group of runsets. 



**Attributes:**
 
 - `runset_name`:  The name of the runset. 
 - `keys`:  The keys to group by. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L91"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroupKey`
A key for grouping runsets. 



**Attributes:**
 
 - `key`:  The metric type to group by. 
 - `value`:  The value of the metric to group by. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1855"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ScalarChart`
A panel object that shows a scalar chart. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `metric` (Required[MetricType]):  The name of a metric logged to your W&B project that the  report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]):  Aggregate runs with specified  function. Options include "mean", "min", "max", "median", "sum", "samples", or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]):   Group runs based on a range. Options  include "minmax", "stddev", "stderr", "none", "samples", or `None`. 
 - `custom_expressions` (Optional[LList[str]]):  A list of custom expressions to be used in the scalar chart. 
 - `legend_template` (Optional[str]):  The template for the legend. 
 - `font_size Optional[FontSize]`:  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1651"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ScatterPlot`
A panel object that shows a 2D or 3D scatter plot. 



**Arguments:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `x Optional[SummaryOrConfigOnlyMetric]`:  The name of a metric logged to your W&B project that the  report pulls information from. The metric specified is used for the x-axis. 
 - `y Optional[SummaryOrConfigOnlyMetric]`:   One or more metrics logged to your W&B project that the report pulls  information from. Metrics specified are plotted within the y-axis. z Optional[SummaryOrConfigOnlyMetric]: 
 - `range_x` (tuple):  Tuple that specifies the range of the x-axis.  
 - `range_y` (tuple):  Tuple that specifies the range of the y-axis.  
 - `range_z`:  Tuple that specifies the range of the z-axis.  
 - `log_x` (Optional[bool]):  Plots the x-coordinates using a base-10 logarithmic scale. 
 - `log_y` (Optional[bool]):  Plots the y-coordinates using a base-10 logarithmic scale. 
 - `log_z` (Optional[bool]):  Plots the z-coordinates using a base-10 logarithmic scale. 
 - `running_ymin` (Optional[bool]):   Apply a moving average or rolling mean on  INSERT. 
 - `running_ymax` (Optional[bool]):  Apply a moving average or rolling mean on INSERT. 
 - `running_ymean` (Optional[bool]):  Apply a moving average or rolling mean on INSERT. 
 - `legend_template` (Optional[str]):   A string that specifies the format of the legend. 
 - `gradient` (Optional[LList[GradientPoint]]):   A list of gradient points that specify the color gradient of the plot. 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `regression` (Optional[bool]):  If `True`, a regression line is plotted on the scatter plot. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L655"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SoundCloud`
A block that renders a SoundCloud player. 



**Attributes:**
 
 - `html`:  The HTML code to embed the SoundCloud player. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L638"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Spotify`
A block that renders a Spotify player. 



**Attributes:**
 
 - `spotify_id`:  The Spotify ID of the track or playlist. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L136"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SummaryMetric`
A summary metric to display in a report. 



**Attributes:**
 
 - `name`:  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L912"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TableOfContents`
A block that contains a list of sections and subsections using H1, H2, and H3 HTML tags specified in a report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L192"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TextWithInlineComments`
A block of text with inline comments. 



**Attributes:**
 
 - `text`:  The text of the block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L926"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Twitter`
A block that displays a Twitter feed. 



**Attributes:**
 
 - `html` (str):  The HTML code to display the Twitter feed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L173"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2268"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownPanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L468"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedList`
A list of items in a bulleted list. 



**Attributes:**
 
 - `items`:  A list of one or more `UnorderedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L403"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedListItem`
A list item in an unordered list. 



**Attributes:**
 
 - `text`:  The text of the list item. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L621"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Video`
A block that renders a video. 



**Attributes:**
 
 - `url`:  The URL of the video. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L943"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1316"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1167"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L947"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockSummaryTable`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2288"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2609"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2488"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2300"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelSummaryTable`








