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

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1732"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

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
 - `custom_expressions` (Optional[LList[str]]):  INSERT. 
 - `legend_template` (Optional[str]):  INSERT 
 - `font_size` ( Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `line_titles` (Optional[dict]):  INSERT. 
 - `line_colors` (Optional[dict]):  INSERT. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L174"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Block`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L475"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `BlockQuote`
An HTML block that indents the specified text as a quotation. 



**Attributes:**
 
 - `text` (str):  Text to render in block quote. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L591"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CalloutBlock`
An HTML block that renders a callout block. 



**Attributes:**
 
 - `text` (str):  The text to render in the callout block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L429"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedList`
An HTML block that contains a list of checkboxes. 



**Attributes:**
 
 - `items` (list):  A list of one or more `CheckedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L361"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedListItem`
A check mark item. Use with `CheckedList` to add one  or more check mark items to your report. 



**Attributes:**
 
 - `text` (str):  The text to render next to the list item. 
 - `checked` (bool):  Add a check mark to the list item. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L494"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeBlock`
An HTML block that contains code samples. 



**Attributes:**
 
 - `code` (str):  A string that contains example code. 
 - `language` (str):  The language the code is written in.  The language specified is used for syntax highlighting.  







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1887"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeComparer`
A panel object that compares the code between two different runs. 



**Attributes:**
 
 - `diff` (Required):  How to display code differences.  Options include "split" and "unified".  







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L128"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Config`
INSERT 



**Attributes:**
 
 - `name` (str):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2131"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CustomChart`
A panel that shows a custom chart. INSERT 



**Attributes:**
 
 - `query` (dict):  INSERT 
 - `chart_name` (str):  The title of the custom chart. 
 - `chart_fields` (dict):  Key-value pairs that define the axis of the  plot. Where the key is the label, and the value is the metric. 
 - `chart_strings` (dict):  INSERT 






---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L687"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Gallery`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L674"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryReport`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L679"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryURL`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1466"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GradientPoint`
INSERT 



**Attributes:**
 
 - `color`:  INSERT 
 - `offset`:  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L230"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H1`
Creates an H1 HTML tag with the text specified. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L248"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H2`
Creates an H2 HTML tag with the text specified. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L266"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H3`
Creates an H3 HTML tag with the text specified. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L212"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Heading`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L614"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `HorizontalRule`
HTML horizontal line. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L563"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Image`
An image HTML block.  



**Attributes:**
 
 - `url` (str):  The URL where your image is hosted. 
 - `caption` (str):  A description of the image that appears underneath the image. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L315"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineCode`
Displays code in a line. Does not add newline character after provided. This differs from `CodeBlock` where the latter creates an HTML block with code. 



**Attributes:**
 
 - `text` (str):  A string that contains example code. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L302"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineLatex`
Displays LaTeX in a line. Does not add newline character after provided LaTeX. This differs from `LatexBlock` where the latter creates an HTML block with LaTeX 



**Attributes:**
 
 - `text` (str):   Text you want to appear in the LaTeX inline.  







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L544"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LatexBlock`
A LaTeX block. Useful if you want to write formulas with LaTeX syntax. 



**Attributes:**
 
 - `text` (str):  Text you want to appear in the LaTeX block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L150"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Layout`
Adjusts the width, height, x-axis, or y-axis of a plot. 



**Attributes:**
 
 - `x` (int):  INSERT 
 - `y` (int):  INSERT 
 - `w` (int):  INSERT 
 - `h` (int):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1486"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LinePlot`
A panel object that shows 2D line plots. 



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
 - `smoothing_show_original` (Optional[bool]):  INSERT. 
 - `max_runs_to_show` (Optional[int]):  The maximum number of runs to show on the line plot. 
 - `custom_expressions` (Optional[LList[str]]):  INSERT. 
 - `plot_type Optional[LinePlotStyle]`:  The type of line plot to generate.  Options include "line", "stacked-area", or "pct-area". 
 - `font_size Optional[FontSize]`:  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `legend_position Optional[LegendPosition]`:  Where to place the legend.  Options include "north", "south", "east", "west", or `None`. 
 - `legend_template` (Optional[str]):  INSERT. 
 - `aggregate` (Optional[bool]):  INSERT. 
 - `xaxis_expression` (Optional[str]):  INSERT. 
 - `legend_fields` (Optional[LList[str]]):  INSERT. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L284"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Link`
Creates a hyperlink. 



**Attributes:**
 
 - `text`:  The text you want to add a hyperlink to. 
 - `url`:  The URL that the hyperlink uses. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L410"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `List`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L348"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ListItem`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L525"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownBlock`
A Markdown block. Useful if you want to write text that uses common markdown syntax. 



**Attributes:**
 
 - `text` (str):  Text you want to appear in the markdown block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2103"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownPanel`
A panel that renders a markdown. 



**Attributes:**
 
 - `markdown` (str):  The text you want to appear in the markdown panel. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2069"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MediaBrowser`
INSERT 



**Attributes:**
 
 - `num_columns` (Optional[int]):  INSERT 
 - `media_keys` (LList[str]):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L117"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Metric`
INSERT 



**Attributes:**
 
 - `name` (str):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L728"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderBy`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L444"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedList`
An HTML block that contains an ordered list. 



**Attributes:**
 
 - `items` (list):  An ordered list of items. Renders as a numbered list. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L385"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedListItem`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L328"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `P`
An HTML paragraph block.  



**Attributes:**
 
 - `text` (str):  The text that appears within the text block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L807"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Panel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L816"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `PanelGrid`
An HTML block where you can add `Runset` and `Panel` objects to your project. 

Available panels include: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`. 





**Attributes:**
 
 - `runsets` (list):  A list of one or more Runset objects. 
 - `panels` (list):  A list of one or more Panel objects. 
 - `active_runset` (int):   The number of runs you want to display within a runset. 
 - `custom_run_colors` (dict):  Key-value pairs where the key is the name of a  run and the value is a color specified by a hexadecimal value. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1951"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlot`
A panel object that shows a parallel coordinates plot. 



**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]):  A list of one  or more `ParallelCoordinatesPlotColumn` objects.  
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `gradient` (Optional[LList[GradientPoint]]):  INSERT. 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1915"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
A column within a parallel coordinates plot.  The order of `metric`s specified  determine the order of the parallel axis (x-axis) in the parallel coordinates plot. 



**Attributes:**
 
 - `metric`:  The name of the metric logged to your W&B project that the report pulls information from. 
 - `display_name` (str):  The name of the metric  inverted (bool): log (bool): 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2005"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParameterImportancePlot`
A panel that shows how important each hyperparameter is in predicting the chosen metric. 



**Attributes:**
 
 - `with_respect_to` (str):  The metric you want to compare the  parameter importance against. Common metrics might include the loss, accuracy,  and so forth. The metric you specify must be logged within the project  that the report pulls information from. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2681"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

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

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2039"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunComparer`
A panel that compares metrics across different runs from the project the report pulls information from. 



**Attributes:**
 
 - `diff_only` (Optional[Literal["split", True]]):  Display only the  difference across runs in a project. You can toggle this feature on and off in the W&B Report UI. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L747"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Runset`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L104"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroup`
UI element that shows runsets.  



**Attributes:**
 
 - `runset_name` (str):  The label of a runset. 
 - `keys` (tuple):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L91"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroupKey`
INSERT 



**Attributes:**
 
 - `key`:  INSERT 
 - `value` (str):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1829"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ScalarChart`
A panel object that shows a scalar chart. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `metric` (Required[MetricType]):  The name of a metric logged to your W&B project that the  report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]):  Aggregate runs with specified  function. Options include "mean", "min", "max", "median", "sum", "samples", or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]):   Group runs based on a range. Options  include "minmax", "stddev", "stderr", "none", "samples", or `None`. 
 - `custom_expressions` (Optional[LList[str]]):  INSERT. 
 - `legend_template` (Optional[str]):  INSERT. 
 - `font_size Optional[FontSize]`:  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1625"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

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
 - `running_ymin` (Optional[bool]):  Apply a moving average or rolling mean on INSERT. 
 - `running_ymax` (Optional[bool]):  Apply a moving average or rolling mean on INSERT. 
 - `running_ymean` (Optional[bool]):  Apply a moving average or rolling mean on INSERT. 
 - `legend_template` (Optional[str]):  INSERT. 
 - `gradient` (Optional[LList[GradientPoint]]):  INSERT. 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `regression` (Optional[bool]):  INSERT. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L662"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SoundCloud`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L643"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Spotify`
An HTML block that renders an iFrame with the song, album, playlist, and so forth. 



**Attributes:**
 
 - `spotify_id` (str):  The base-62 identifier found at the end  of the Spotify URI for an artist, track, album, playlist, and so forth. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L139"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SummaryMetric`
INSERT 



**Attributes:**
 
 - `name` (str):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L891"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TableOfContents`
An HTML block that contains a list of sections and subsections using H1, H2, and H3 HTML tags specified in a report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L197"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TextWithInlineComments`
INSERT 



**Attributes:**
 
 - `text` (str):  INSERT 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L905"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Twitter`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L178"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2242"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownPanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L459"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedList`
An HTML block that contains an unordered list. 



**Attributes:**
 
 - `items` (list):  An unordered list of items.  Renders as bullet points (small black circles). 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L398"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedListItem`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L625"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Video`
An HTML block that renders a video. 



**Attributes:**
 
 - `url` (str):  The URL where the video is hosted. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L917"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1290"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1141"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L921"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockSummaryTable`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2262"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2583"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2462"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2274"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelSummaryTable`








