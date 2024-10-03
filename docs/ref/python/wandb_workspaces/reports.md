# Reports

<!-- markdownlint-disable -->

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2.py#L0"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
Python library for programmatically working with W&B Reports API. 

```python
# How to import
import wandb_workspaces.reports.v2
```

---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1843"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `BarPlot`
A panel object that shows a 2D bar plot. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. metrics LList[MetricType]: 
 - `orientation Literal["v", "h"]`:  The orientation of the bar plot.  Set to either vertical ("v") or horizontal ("h"). Defaults to horizontal ("h"). 
 - `range_x` (Tuple[float | None, float | None]):  Tuple that specifies the range of the x-axis. 
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

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L186"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Block`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L514"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `BlockQuote`
A block of quoted text. 



**Attributes:**
 
 - `text` (str):  The text of the block quote. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L627"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CalloutBlock`
A block of callout text. 



**Attributes:**
 
 - `text` (str):  The callout text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L469"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedList`
A list of items with checkboxes. Add one or more `CheckedListItem` within `CheckedList`. 



**Attributes:**
 
 - `items` (LList[CheckedListItem]):  A list of one or more `CheckedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L391"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CheckedListItem`
A list item with a checkbox. Add one or more `CheckedListItem` within `CheckedList`. 



**Attributes:**
 
 - `text` (str):  The text of the list item. 
 - `checked` (bool):  Whether the checkbox is checked. By default, set to `False`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L532"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeBlock`
A block of code. 



**Attributes:**
 
 - `code` (str):  The code in the block. 
 - `language` (Optional[Language]):  The language of the code. Language specified  is used for syntax highlighting. By default, set to "python". Options include  'javascript', 'python', 'css', 'json', 'html', 'markdown', 'yaml'. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1998"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CodeComparer`
A panel object that compares the code between two different runs. 



**Attributes:**
 
 - `diff` (Literal['split', 'unified']):  How to display code differences.  Options include "split" and "unified". 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L135"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Config`
Metrics logged to a run's config object. Config objects are commonly logged using `run.config[name] = ...` or passing a config as a dictionary of key-value pairs, where the key is the name of the metric and the value is the value of that metric.  



**Attributes:**
 
 - `name` (str):  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2250"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `CustomChart`
A panel that shows a custom chart. The chart is defined by a weave query. 



**Attributes:**
 
 - `query` (dict):  The query that defines the custom chart. The key is the name of the field, and the value is the query. 
 - `chart_name` (str):  The title of the custom chart. 
 - `chart_fields` (dict):  Key-value pairs that define the axis of the  plot. Where the key is the label, and the value is the metric. 
 - `chart_strings` (dict):  Key-value pairs that define the strings in the chart. 




---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2270"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>classmethod</kbd> `from_table`

```python
from_table(
    table_name: str,
    chart_fields: dict = None,
    chart_strings: dict = None
)
```

Create a custom chart from a table. 



**Arguments:**
 
 - `table_name` (str):  The name of the table. 
 - `chart_fields` (dict):  The fields to display in the chart.  
 - `chart_strings` (dict):  The strings to display in the chart.  




---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L742"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Gallery`
A block that renders a gallery of reports and URLs. 



**Attributes:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]):  A list of  `GalleryReport` and `GalleryURL` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L716"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryReport`
A reference to a report in the gallery. 



**Attributes:**
 
 - `report_id` (str):  The ID of the report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L726"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GalleryURL`
A URL to an external resource. 



**Attributes:**
 
 - `url` (str):  The URL of the resource. 
 - `title` (Optional[str]):  The title of the resource. 
 - `description` (Optional[str]):  The description of the resource. 
 - `image_url` (Optional[str]):  The URL of an image to display. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1575"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `GradientPoint`
A point in a gradient. 



**Attributes:**
 
 - `color`:  The color of the point. 
 - `offset`:  The position of the point in the gradient. The value should be between 0 and 100. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L243"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H1`
An H1 heading with the text specified. 



**Attributes:**
 
 - `text` (str):  The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]):  The blocks to show when the heading is collapsed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L267"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H2`
An H2 heading with the text specified. 



**Attributes:**
 
 - `text` (str):  The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]):  One or more blocks to  show when the heading is collapsed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L292"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `H3`
An H3 heading with the text specified. 



**Attributes:**
 
 - `text` (str):  The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]):  One or more blocks to  show when the heading is collapsed.  







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L224"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Heading`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L650"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `HorizontalRule`
HTML horizontal line. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L600"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Image`
A block that renders an image. 



**Attributes:**
 
 - `url` (str):  The URL of the image. 
 - `caption` (str):  The caption of the image.  Caption appears underneath the image. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L346"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineCode`
Inline code. Does not add newline character after code. 



**Attributes:**
 
 - `text` (str):  The code you want to appear in the report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L334"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `InlineLatex`
Inline LaTeX markdown. Does not add newline character after the LaTeX markdown. 



**Attributes:**
 
 - `text` (str):  LaTeX markdown you want to appear in the report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L582"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LatexBlock`
A block of LaTeX text. 



**Attributes:**
 
 - `text` (str):  The LaTeX text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L162"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Layout`
The layout of a panel in a report. Adjusts the size and position of the panel. 



**Attributes:**
 
 - `x` (int):  The x position of the panel. 
 - `y` (int):  The y position of the panel. 
 - `w` (int):  The width of the panel. 
 - `h` (int):  The height of the panel. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1596"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `LinePlot`
A panel object with 2D line plots. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `x` (Optional[MetricType]):  The name of a metric logged to your W&B project that the  report pulls information from. The metric specified is used for the x-axis. 
 - `y` (LList[MetricType]):  One or more metrics logged to your W&B project that the report pulls  information from. The metric specified is used for the y-axis. 
 - `range_x` (Tuple[float | `None`, float | `None`]):  Tuple that specifies the range of the x-axis. 
 - `range_y` (Tuple[float | `None`, float | `None`]):  Tuple that specifies the range of the y-axis. 
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

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L317"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Link`
A link to a URL. 



**Attributes:**
 
 - `text` (Union[str, TextWithInlineComments]):  The text of the link. 
 - `url` (str):  The URL the link points to. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L449"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `List`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L378"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ListItem`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L563"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownBlock`
A block of markdown text. Useful if you want to write text that uses common markdown syntax. 



**Attributes:**
 
 - `text` (str):  The markdown text. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2221"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MarkdownPanel`
A panel that renders markdown. 



**Attributes:**
 
 - `markdown` (str):  The text you want to appear in the markdown panel. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2186"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `MediaBrowser`
A panel that displays media files in a grid layout. 



**Attributes:**
 
 - `num_columns` (Optional[int]):  The number of columns in the grid. 
 - `media_keys` (LList[str]):  A list of media keys that correspond to the media files. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L122"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Metric`
A metric to display in a report that is logged in your project. 



**Attributes:**
 
 - `name` (str):  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L790"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderBy`
A metric to order by. 



**Attributes:**
 
 - `name` (str):  The name of the metric. 
 - `ascending` (bool):  Whether to sort in ascending order.  By default set to `False`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L484"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedList`
A list of items in a numbered list. 



**Attributes:**
 
 - `items` (LList[str]):  A list of one or more `OrderedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L412"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `OrderedListItem`
A list item in an ordered list. 



**Attributes:**
 
 - `text` (str):  The text of the list item. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L358"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `P`
A paragraph of text. 



**Attributes:**
 
 - `text` (str):  The text of the paragraph. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L890"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Panel`
A panel that displays a visualization in a panel grid. 



**Attributes:**
 
 - `layout` (Layout):  A `Layout` object. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L905"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `PanelGrid`
A grid that consists of runsets and panels. Add runsets and panels with `Runset` and `Panel` objects, respectively. 

Available panels include: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`. 





**Attributes:**
 
 - `runsets` (LList["Runset"]):  A list of one or more `Runset` objects. 
 - `panels` (LList["PanelTypes"]):  A list of one or more `Panel` objects. 
 - `active_runset` (int):  The number of runs you want to display within a runset. By default, it is set to 0. 
 - `custom_run_colors` (dict):  Key-value pairs where the key is the name of a  run and the value is a color specified by a hexadecimal value. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2065"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlot`
A panel object that shows a parallel coordinates plot. 



**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]):  A list of one  or more `ParallelCoordinatesPlotColumn` objects. 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `gradient` (Optional[LList[GradientPoint]]):  A list of gradient points. 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2027"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
A column within a parallel coordinates plot.  The order of `metric`s specified determine the order of the parallel axis (x-axis) in the parallel coordinates plot. 



**Attributes:**
 
 - `metric` (str | Config | SummaryMetric):  The name of the  metric logged to your W&B project that the report pulls information from. 
 - `display_name` (Optional[str]):  The name of the metric 
 - `inverted` (Optional[bool]):  Whether to invert the metric. 
 - `log` (Optional[bool]):  Whether to apply a log transformation to the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2120"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ParameterImportancePlot`
A panel that shows how important each hyperparameter is in predicting the chosen metric. 



**Attributes:**
 
 - `with_respect_to` (str):  The metric you want to compare the  parameter importance against. Common metrics might include the loss, accuracy,  and so forth. The metric you specify must be logged within the project  that the report pulls information from. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2801"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Report`
An object that represents a W&B Report. Use the returned object's `blocks` attribute to customize your report. Report objects do not automatically save. Use the `save()` method to persists changes. 



**Attributes:**
 
 - `project` (str):  The name of the W&B project you want to load in. The project specified appears in the report's URL. 
 - `entity` (str):  The W&B entity that owns the report. The entity appears in the report's URL. 
 - `title` (str):  The title of the report. The title appears at the top of the report as an H1 heading. 
 - `description` (str):  A description of the report. The description appears underneath the report's title. 
 - `blocks` (LList[BlockTypes]):  A list of one or more HTML tags, plots, grids, runsets, or more. 
 - `width` (Literal['readable', 'fixed', 'fluid']):  The width of the report. Options include 'readable', 'fixed', 'fluid'. 


---

#### <kbd>property</kbd> url

The URL where the report is hosted. The report URL consists of `https://wandb.ai/{entity}/{project_name}/reports/`. Where `{entity}` and `{project_name}` consists of the entity that the report belongs to and the name of the project, respectively. 



---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2949"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

Load in the report into current environment. Pass in the URL where the report is hosted. 



**Arguments:**
 
 - `url` (str):  The URL where the report is hosted. 
 - `as_model` (bool):  If True, return the model object instead of the Report object.  By default, set to `False`. 

---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2911"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Persists changes made to a report object. 

---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2965"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

Generate HTML containing an iframe displaying this report. Commonly used to within a Python notebook.  



**Arguments:**
 
 - `height` (int):  Height of the iframe. 
 - `hidden` (bool):  If True, hide the iframe. Default set to `False`.

---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2155"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunComparer`
A panel that compares metrics across different runs from the project the report pulls information from. 



**Attributes:**
 
 - `diff_only` (Optional[Literal["split", `True`]]):  Display only the  difference across runs in a project. You can toggle this feature on and off in the W&B Report UI. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L817"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Runset`
A set of runs to display in a panel grid. 



**Attributes:**
 
 - `entity` (str):  The entity name. 
 - `project` (str):  The project name. 
 - `name` (str):  The name of the run set. Set to `Run set` by default. 
 - `query` (str):  A query string to filter runs. 
 - `filters` (Optional[str]):  A filter string to filter runs. 
 - `groupby` (LList[str]):  A list of metric names to group by. 
 - `order` (LList[OrderBy]):  A list of `OrderBy` objects to order by. 
 - `custom_run_colors` (LList[OrderBy]):  A dictionary mapping run IDs to colors. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L106"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroup`
UI element that shows a group of runsets.  



**Attributes:**
 
 - `runset_name` (str):  The name of the runset. 
 - `keys` (str):  The keys to group by.  Pass in one or more `RunsetGroupKey`  objects to group by. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L91"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `RunsetGroupKey`
Groups runsets by a metric type and value. Part of a `RunsetGroup`. Specify the metric type and value to group by as key-value pairs. 



**Attributes:**
 
 - `key` (str):  The metric type to group by. 
 - `value` (str):  The value of the metric to group by. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1939"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ScalarChart`
A panel object that shows a scalar chart. 



**Attributes:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `metric` (MetricType):  The name of a metric logged to your W&B project that the  report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]):  Aggregate runs with specified  function. Options include "mean", "min", "max", "median", "sum", "samples", or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]):   Group runs based on a range. Options  include "minmax", "stddev", "stderr", "none", "samples", or `None`. 
 - `custom_expressions` (Optional[LList[str]]):  A list of custom expressions to be used in the scalar chart. 
 - `legend_template` (Optional[str]):  The template for the legend. 
 - `font_size Optional[FontSize]`:  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1735"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `ScatterPlot`
A panel object that shows a 2D or 3D scatter plot. 



**Arguments:**
 
 - `title` (Optional[str]):  The text that appears at the top of the plot. 
 - `x Optional[SummaryOrConfigOnlyMetric]`:  The name of a metric logged to your W&B project that the  report pulls information from. The metric specified is used for the x-axis. 
 - `y Optional[SummaryOrConfigOnlyMetric]`:   One or more metrics logged to your W&B project that the report pulls  information from. Metrics specified are plotted within the y-axis. z Optional[SummaryOrConfigOnlyMetric]: 
 - `range_x` (Tuple[float | `None`, float | `None`]):  Tuple that specifies the range of the x-axis. 
 - `range_y` (Tuple[float | `None`, float | `None`]):  Tuple that specifies the range of the y-axis. 
 - `range_z` (Tuple[float | `None`, float | `None`]):  Tuple that specifies the range of the z-axis. 
 - `log_x` (Optional[bool]):  Plots the x-coordinates using a base-10 logarithmic scale. 
 - `log_y` (Optional[bool]):  Plots the y-coordinates using a base-10 logarithmic scale. 
 - `log_z` (Optional[bool]):  Plots the z-coordinates using a base-10 logarithmic scale. 
 - `running_ymin` (Optional[bool]):   Apply a moving average or rolling mean. 
 - `running_ymax` (Optional[bool]):  Apply a moving average or rolling mean. 
 - `running_ymean` (Optional[bool]):  Apply a moving average or rolling mean. 
 - `legend_template` (Optional[str]):   A string that specifies the format of the legend. 
 - `gradient` (Optional[LList[GradientPoint]]):   A list of gradient points that specify the color gradient of the plot. 
 - `font_size` (Optional[FontSize]):  The size of the line plot's font.  Options include "small", "medium", "large", "auto", or `None`. 
 - `regression` (Optional[bool]):  If `True`, a regression line is plotted on the scatter plot. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L698"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SoundCloud`
A block that renders a SoundCloud player. 



**Attributes:**
 
 - `html` (str):  The HTML code to embed the SoundCloud player. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L680"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Spotify`
A block that renders a Spotify player. 



**Attributes:**
 
 - `spotify_id` (str):  The Spotify ID of the track or playlist. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L151"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `SummaryMetric`
A summary metric to display in a report. 



**Attributes:**
 
 - `name` (str):  The name of the metric. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L982"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TableOfContents`
A block that contains a list of sections and subsections using H1, H2, and H3 HTML blocks specified in a report. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L209"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `TextWithInlineComments`
A block of text with inline comments. 



**Attributes:**
 
 - `text` (str):  The text of the block. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L997"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Twitter`
A block that displays a Twitter feed. 



**Attributes:**
 
 - `html` (str):  The HTML code to display the Twitter feed. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L190"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2362"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnknownPanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L499"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedList`
A list of items in a bulleted list. 



**Attributes:**
 
 - `items` (LList[str]):  A list of one or more `UnorderedListItem` objects. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L431"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `UnorderedListItem`
A list item in an unordered list. 



**Attributes:**
 
 - `text` (str):  The text of the list item. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L662"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `Video`
A block that renders a video. 



**Attributes:**
 
 - `url` (str):  The URL of the video. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1015"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlock`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1399"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1250"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L1019"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeaveBlockSummaryTable`
A block that displays a summary table of a query panel. 

The term "Weave" in this API does not refer to the W&B Weave toolkit used for tracking and evaluating LLM.  





**Attributes:**
 
 - `entity` (str):  The entity name. 
 - `project` (str):  The project name. 
 - `table_name` (str):  The table name. 







---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2382"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanel`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2703"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifact`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2582"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`










---

<a href="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py#L2394"><img align="right" src="https://img.shields.io/badge/-source-cccccc?style=flat-square" /></a>

## <kbd>class</kbd> `WeavePanelSummaryTable`








