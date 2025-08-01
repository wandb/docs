---
title: Reports
---
{{< cta-button githubLink="https://github.com/wandb/wandb-workspaces/blob/main/wandb_workspaces/reports/v2/interface.py" >}}

<!-- markdownlint-turnedoff -->

{{% alert %}}
W&B Report and Workspace API is in Public Preview.
{{% /alert %}}

# <kbd>module</kbd> `wandb_workspaces.reports.v2`
Python library for programmatically working with W&B Reports API. 

```python
import wandb_workspaces.reports.v2 as wr

report = wr.Report(
     entity="entity",
     project="project",
     title="An amazing title",
     description="A descriptive description.",
)

blocks = [
     wr.PanelGrid(
         panels=[
             wr.LinePlot(x="time", y="velocity"),
             wr.ScatterPlot(x="time", y="acceleration"),
         ]
     )
]

report.blocks = blocks
report.save()
```

---



## <kbd>class</kbd> `BarPlot`
A panel object that shows a 2D bar plot. 



**Attributes:**
 
 - `title` (Optional[str]): The text that appears at the top of the plot. 
 - `metrics` (LList[MetricType]): orientation Literal["v", "h"]: The orientation of the bar plot. Set to either vertical ("v") or horizontal ("h"). Defaults to horizontal ("h"). 
 - `range_x` (Tuple[float | None, float | None]): Tuple that specifies the range of the x-axis. 
 - `title_x` (Optional[str]): The label of the x-axis. 
 - `title_y` (Optional[str]): The label of the y-axis. 
 - `groupby` (Optional[str]): Group runs based on a metric logged to your W&B project that the report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]): Aggregate runs with specified function. Options include `mean`, `min`, `max`, `median`, `sum`, `samples`, or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]): Group runs based on a range. Options include `minmax`, `stddev`, `stderr`, `none`, =`samples`, or `None`. 
 - `max_runs_to_show` (Optional[int]): The maximum number of runs to show on the plot. 
 - `max_bars_to_show` (Optional[int]): The maximum number of bars to show on the bar plot. 
 - `custom_expressions` (Optional[LList[str]]): A list of custom expressions to be used in the bar plot. 
 - `legend_template` (Optional[str]): The template for the legend. 
 - `font_size` ( Optional[FontSize]): The size of the line plot's font. Options include `small`, `medium`, `large`, `auto`, or `None`. 
 - `line_titles` (Optional[dict]): The titles of the lines. The keys are the line names and the values are the titles. 
 - `line_colors` (Optional[dict]): The colors of the lines. The keys are the line names and the values are the colors. 







---



## <kbd>class</kbd> `BlockQuote`
A block of quoted text. 



**Attributes:**
 
 - `text` (str): The text of the block quote. 







---



## <kbd>class</kbd> `CalloutBlock`
A block of callout text. 



**Attributes:**
 
 - `text` (str): The callout text. 







---



## <kbd>class</kbd> `CheckedList`
A list of items with checkboxes. Add one or more `CheckedListItem` within `CheckedList`. 



**Attributes:**
 
 - `items` (LList[CheckedListItem]): A list of one or more `CheckedListItem` objects. 







---



## <kbd>class</kbd> `CheckedListItem`
A list item with a checkbox. Add one or more `CheckedListItem` within `CheckedList`. 



**Attributes:**
 
 - `text` (str): The text of the list item. 
 - `checked` (bool): Whether the checkbox is checked. By default, set to `False`. 







---



## <kbd>class</kbd> `CodeBlock`
A block of code. 



**Attributes:**
 
 - `code` (str): The code in the block. 
 - `language` (Optional[Language]): The language of the code. Language specified is used for syntax highlighting. By default, set to `python`. Options include `javascript`, `python`, `css`, `json`, `html`, `markdown`, `yaml`. 







---



## <kbd>class</kbd> `CodeComparer`
A panel object that compares the code between two different runs. 



**Attributes:**
 
 - `diff` `(Literal['split', 'unified'])`: How to display code differences. Options include `split` and `unified`. 







---



## <kbd>class</kbd> `Config`
Metrics logged to a run's config object. Config objects are commonly logged using `wandb.Run.config[name] = ...` or passing a config as a dictionary of key-value pairs, where the key is the name of the metric and the value is the value of that metric. 



**Attributes:**
 
 - `name` (str): The name of the metric. 







---



## <kbd>class</kbd> `CustomChart`
A panel that shows a custom chart. The chart is defined by a weave query. 



**Attributes:**
 
 - `query` (dict): The query that defines the custom chart. The key is the name of the field, and the value is the query. 
 - `chart_name` (str): The title of the custom chart. 
 - `chart_fields` (dict): Key-value pairs that define the axis of the plot. Where the key is the label, and the value is the metric. 
 - `chart_strings` (dict): Key-value pairs that define the strings in the chart. 




---



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
 
 - `table_name` (str): The name of the table. 
 - `chart_fields` (dict): The fields to display in the chart. 
 - `chart_strings` (dict): The strings to display in the chart. 




---



## <kbd>class</kbd> `Gallery`
A block that renders a gallery of reports and URLs. 



**Attributes:**
 
 - `items` (List[Union[`GalleryReport`, `GalleryURL`]]): A list of `GalleryReport` and `GalleryURL` objects. 







---



## <kbd>class</kbd> `GalleryReport`
A reference to a report in the gallery. 



**Attributes:**
 
 - `report_id` (str): The ID of the report. 







---



## <kbd>class</kbd> `GalleryURL`
A URL to an external resource. 



**Attributes:**
 
 - `url` (str): The URL of the resource. 
 - `title` (Optional[str]): The title of the resource. 
 - `description` (Optional[str]): The description of the resource. 
 - `image_url` (Optional[str]): The URL of an image to display. 







---



## <kbd>class</kbd> `GradientPoint`
A point in a gradient. 



**Attributes:**
 
 - `color`: The color of the point. 
 - `offset`: The position of the point in the gradient. The value should be between 0 and 100. 







---



## <kbd>class</kbd> `H1`
An H1 heading with the text specified. 



**Attributes:**
 
 - `text` (str): The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): The blocks to show when the heading is collapsed. 







---



## <kbd>class</kbd> `H2`
An H2 heading with the text specified. 



**Attributes:**
 
 - `text` (str): The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): One or more blocks to show when the heading is collapsed. 







---



## <kbd>class</kbd> `H3`
An H3 heading with the text specified. 



**Attributes:**
 
 - `text` (str): The text of the heading. 
 - `collapsed_blocks` (Optional[LList["BlockTypes"]]): One or more blocks to show when the heading is collapsed. 







---



## <kbd>class</kbd> `Heading`










---



## <kbd>class</kbd> `HorizontalRule`
HTML horizontal line. 







---



## <kbd>class</kbd> `Image`
A block that renders an image. 



**Attributes:**
 
 - `url` (str): The URL of the image. 
 - `caption` (str): The caption of the image. Caption appears underneath the image. 







---



## <kbd>class</kbd> `InlineCode`
Inline code. Does not add newline character after code. 



**Attributes:**
 
 - `text` (str): The code you want to appear in the report. 







---



## <kbd>class</kbd> `InlineLatex`
Inline LaTeX markdown. Does not add newline character after the LaTeX markdown. 



**Attributes:**
 
 - `text` (str): LaTeX markdown you want to appear in the report. 







---



## <kbd>class</kbd> `LatexBlock`
A block of LaTeX text. 



**Attributes:**
 
 - `text` (str): The LaTeX text. 







---



## <kbd>class</kbd> `Layout`
The layout of a panel in a report. Adjusts the size and position of the panel. 



**Attributes:**
 
 - `x` (int): The x position of the panel. 
 - `y` (int): The y position of the panel. 
 - `w` (int): The width of the panel. 
 - `h` (int): The height of the panel. 







---



## <kbd>class</kbd> `LinePlot`
A panel object with 2D line plots. 



**Attributes:**
 
 - `title` (Optional[str]): The text that appears at the top of the plot. 
 - `x` (Optional[MetricType]): The name of a metric logged to your W&B project that the report pulls information from. The metric specified is used for the x-axis. 
 - `y` (LList[MetricType]): One or more metrics logged to your W&B project that the report pulls information from. The metric specified is used for the y-axis. 
 - `range_x` (Tuple[float | `None`, float | `None`]): Tuple that specifies the range of the x-axis. 
 - `range_y` (Tuple[float | `None`, float | `None`]): Tuple that specifies the range of the y-axis. 
 - `log_x` (Optional[bool]): Plots the x-coordinates using a base-10 logarithmic scale. 
 - `log_y` (Optional[bool]): Plots the y-coordinates using a base-10 logarithmic scale. 
 - `title_x` (Optional[str]): The label of the x-axis. 
 - `title_y` (Optional[str]): The label of the y-axis. 
 - `ignore_outliers` (Optional[bool]): If set to `True`, do not plot outliers. 
 - `groupby` (Optional[str]): Group runs based on a metric logged to your W&B project that the report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]): Aggregate runs with specified function. Options include `mean`, `min`, `max`, `median`, `sum`, `samples`, or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]): Group runs based on a range. Options include `minmax`, `stddev`, `stderr`, `none`, `samples`, or `None`. 
 - `smoothing_factor` (Optional[float]): The smoothing factor to apply to the smoothing type. Accepted values range between 0 and 1. 
 - `smoothing_type Optional[SmoothingType]`: Apply a filter based on the specified distribution. Options include `exponentialTimeWeighted`, `exponential`, `gaussian`, `average`, or `none`. 
 - `smoothing_show_original` (Optional[bool]): If set to `True`, show the original data. 
 - `max_runs_to_show` (Optional[int]): The maximum number of runs to show on the line plot. 
 - `custom_expressions` (Optional[LList[str]]): Custom expressions to apply to the data. 
 - `plot_type Optional[LinePlotStyle]`: The type of line plot to generate. Options include `line`, `stacked-area`, or `pct-area`. 
 - `font_size Optional[FontSize]`: The size of the line plot's font. Options include `small`, `medium`, `large`, `auto`, or `None`. 
 - `legend_position Optional[LegendPosition]`: Where to place the legend. Options include `north`, `south`, `east`, `west`, or `None`. 
 - `legend_template` (Optional[str]): The template for the legend. 
 - `aggregate` (Optional[bool]): If set to `True`, aggregate the data. 
 - `xaxis_expression` (Optional[str]): The expression for the x-axis. 
 - `legend_fields` (Optional[LList[str]]): The fields to include in the legend. 







---



## <kbd>class</kbd> `Link`
A link to a URL. 



**Attributes:**
 
 - `text` (Union[str, TextWithInlineComments]): The text of the link. 
 - `url` (str): The URL the link points to. 







---



## <kbd>class</kbd> `MarkdownBlock`
A block of markdown text. Useful if you want to write text that uses common markdown syntax. 



**Attributes:**
 
 - `text` (str): The markdown text. 







---



## <kbd>class</kbd> `MarkdownPanel`
A panel that renders markdown. 



**Attributes:**
 
 - `markdown` (str): The text you want to appear in the markdown panel. 







---



## <kbd>class</kbd> `MediaBrowser`
A panel that displays media files in a grid layout. 



**Attributes:**
 
 - `num_columns` (Optional[int]): The number of columns in the grid. 
 - `media_keys` (LList[str]): A list of media keys that correspond to the media files. 







---



## <kbd>class</kbd> `Metric`
A metric to display in a report that is logged in your project. 



**Attributes:**
 
 - `name` (str): The name of the metric. 







---



## <kbd>class</kbd> `OrderBy`
A metric to order by. 



**Attributes:**
 
 - `name` (str): The name of the metric. 
 - `ascending` (bool): Whether to sort in ascending order. By default set to `False`. 







---



## <kbd>class</kbd> `OrderedList`
A list of items in a numbered list. 



**Attributes:**
 
 - `items` (LList[str]): A list of one or more `OrderedListItem` objects. 







---



## <kbd>class</kbd> `OrderedListItem`
A list item in an ordered list. 



**Attributes:**
 
 - `text` (str): The text of the list item. 







---



## <kbd>class</kbd> `P`
A paragraph of text. 



**Attributes:**
 
 - `text` (str): The text of the paragraph. 







---



## <kbd>class</kbd> `Panel`
A panel that displays a visualization in a panel grid. 



**Attributes:**
 
 - `layout` (Layout): A `Layout` object. 







---



## <kbd>class</kbd> `PanelGrid`
A grid that consists of runsets and panels. Add runsets and panels with `Runset` and `Panel` objects, respectively. 

Available panels include: `LinePlot`, `ScatterPlot`, `BarPlot`, `ScalarChart`, `CodeComparer`, `ParallelCoordinatesPlot`, `ParameterImportancePlot`, `RunComparer`, `MediaBrowser`, `MarkdownPanel`, `CustomChart`, `WeavePanel`, `WeavePanelSummaryTable`, `WeavePanelArtifactVersionedFile`. 





**Attributes:**
 
 - `runsets` (LList["Runset"]): A list of one or more `Runset` objects. 
 - `panels` (LList["PanelTypes"]): A list of one or more `Panel` objects. 
 - `active_runset` (int): The number of runs you want to display within a runset. By default, it is set to 0. 
 - `custom_run_colors` (dict): Key-value pairs where the key is the name of a run and the value is a color specified by a hexadecimal value. 







---



## <kbd>class</kbd> `ParallelCoordinatesPlot`
A panel object that shows a parallel coordinates plot. 



**Attributes:**
 
 - `columns` (LList[ParallelCoordinatesPlotColumn]): A list of one or more `ParallelCoordinatesPlotColumn` objects. 
 - `title` (Optional[str]): The text that appears at the top of the plot. 
 - `gradient` (Optional[LList[GradientPoint]]): A list of gradient points. 
 - `font_size` (Optional[FontSize]): The size of the line plot's font. Options include `small`, `medium`, `large`, `auto`, or `None`. 







---



## <kbd>class</kbd> `ParallelCoordinatesPlotColumn`
A column within a parallel coordinates plot. The order of `metric`s specified determine the order of the parallel axis (x-axis) in the parallel coordinates plot. 



**Attributes:**
 
 - `metric` (str | Config | SummaryMetric): The name of the metric logged to your W&B project that the report pulls information from. 
 - `display_name` (Optional[str]): The name of the metric 
 - `inverted` (Optional[bool]): Whether to invert the metric. 
 - `log` (Optional[bool]): Whether to apply a log transformation to the metric. 







---



## <kbd>class</kbd> `ParameterImportancePlot`
A panel that shows how important each hyperparameter is in predicting the chosen metric. 



**Attributes:**
 
 - `with_respect_to` (str): The metric you want to compare the parameter importance against. Common metrics might include the loss, accuracy, and so forth. The metric you specify must be logged within the project that the report pulls information from. 







---



## <kbd>class</kbd> `Report`
An object that represents a W&B Report. Use the returned object's `blocks` attribute to customize your report. Report objects do not automatically save. Use the `save()` method to persists changes. 



**Attributes:**
 
 - `project` (str): The name of the W&B project you want to load in. The project specified appears in the report's URL. 
 - `entity` (str): The W&B entity that owns the report. The entity appears in the report's URL. 
 - `title` (str): The title of the report. The title appears at the top of the report as an H1 heading. 
 - `description` (str): A description of the report. The description appears underneath the report's title. 
 - `blocks` (LList[BlockTypes]): A list of one or more HTML tags, plots, grids, runsets, and more. 
 - `width` (Literal['readable', 'fixed', 'fluid']): The width of the report. Options include 'readable', 'fixed', 'fluid'. 


---

#### <kbd>property</kbd> url

The URL where the report is hosted. The report URL consists of `https://wandb.ai/{entity}/{project_name}/reports/`. Where `{entity}` and `{project_name}` consists of the entity that the report belongs to and the name of the project, respectively. 



---



### <kbd>classmethod</kbd> `from_url`

```python
from_url(url: str, as_model: bool = False)
```

Load in the report into current environment. Pass in the URL where the report is hosted. 



**Arguments:**
 
 - `url` (str): The URL where the report is hosted. 
 - `as_model` (bool): If True, return the model object instead of the Report object. By default, set to `False`. 

---



### <kbd>method</kbd> `save`

```python
save(draft: bool = False, clone: bool = False)
```

Persists changes made to a report object. 

---



### <kbd>method</kbd> `to_html`

```python
to_html(height: int = 1024, hidden: bool = False) → str
```

Generate HTML containing an iframe displaying this report. Commonly used to within a Python notebook. 



**Arguments:**
 
 - `height` (int): Height of the iframe. 
 - `hidden` (bool): If True, hide the iframe. Default set to `False`.

---



## <kbd>class</kbd> `RunComparer`
A panel that compares metrics across different runs from the project the report pulls information from. 



**Attributes:**
 
 - `diff_only` `(Optional[Literal["split", True]])`: Display only the difference across runs in a project. You can toggle this feature on and off in the W&B Report UI. 







---



## <kbd>class</kbd> `Runset`
A set of runs to display in a panel grid. 



**Attributes:**
 
 - `entity` (str): An entity that owns or has the correct permissions to the project where the runs are stored. 
 - `project` (str): The name of the project were the runs are stored. 
 - `name` (str): The name of the run set. Set to `Run set` by default. 
 - `query` (str): A query string to filter runs. 
 - `filters` (Optional[str]): A filter string to filter runs. 
 - `groupby` (LList[str]): A list of metric names to group by. 
 - `order` (LList[OrderBy]): A list of `OrderBy` objects to order by. 
 - `custom_run_colors` (LList[OrderBy]): A dictionary mapping run IDs to colors. 







---



## <kbd>class</kbd> `RunsetGroup`
UI element that shows a group of runsets. 



**Attributes:**
 
 - `runset_name` (str): The name of the runset. 
 - `keys` (Tuple[RunsetGroupKey, ...]): The keys to group by. Pass in one or more `RunsetGroupKey` objects to group by. 







---



## <kbd>class</kbd> `RunsetGroupKey`
Groups runsets by a metric type and value. Part of a `RunsetGroup`. Specify the metric type and value to group by as key-value pairs. 



**Attributes:**
 
 - `key` (Type[str] | Type[Config] | Type[SummaryMetric] | Type[Metric]): The metric type to group by. 
 - `value` (str): The value of the metric to group by. 







---



## <kbd>class</kbd> `ScalarChart`
A panel object that shows a scalar chart. 



**Attributes:**
 
 - `title` (Optional[str]): The text that appears at the top of the plot. 
 - `metric` (MetricType): The name of a metric logged to your W&B project that the report pulls information from. 
 - `groupby_aggfunc` (Optional[GroupAgg]): Aggregate runs with specified function. Options include `mean`, `min`, `max`, `median`, `sum`, `samples`, or `None`. 
 - `groupby_rangefunc` (Optional[GroupArea]): Group runs based on a range. Options include `minmax`, `stddev`, `stderr`, `none`, `samples`, or `None`. 
 - `custom_expressions` (Optional[LList[str]]): A list of custom expressions to be used in the scalar chart. 
 - `legend_template` (Optional[str]): The template for the legend. 
 - `font_size Optional[FontSize]`: The size of the line plot's font. Options include `small`, `medium`, `large`, `auto`, or `None`. 







---



## <kbd>class</kbd> `ScatterPlot`
A panel object that shows a 2D or 3D scatter plot. 



**Arguments:**
 
 - `title` (Optional[str]): The text that appears at the top of the plot. 
 - `x Optional[SummaryOrConfigOnlyMetric]`: The name of a metric logged to your W&B project that the report pulls information from. The metric specified is used for the x-axis. 
 - `y Optional[SummaryOrConfigOnlyMetric]`: One or more metrics logged to your W&B project that the report pulls information from. Metrics specified are plotted within the y-axis. z Optional[SummaryOrConfigOnlyMetric]: 
 - `range_x` (Tuple[float | `None`, float | `None`]): Tuple that specifies the range of the x-axis. 
 - `range_y` (Tuple[float | `None`, float | `None`]): Tuple that specifies the range of the y-axis. 
 - `range_z` (Tuple[float | `None`, float | `None`]): Tuple that specifies the range of the z-axis. 
 - `log_x` (Optional[bool]): Plots the x-coordinates using a base-10 logarithmic scale. 
 - `log_y` (Optional[bool]): Plots the y-coordinates using a base-10 logarithmic scale. 
 - `log_z` (Optional[bool]): Plots the z-coordinates using a base-10 logarithmic scale. 
 - `running_ymin` (Optional[bool]): Apply a moving average or rolling mean. 
 - `running_ymax` (Optional[bool]): Apply a moving average or rolling mean. 
 - `running_ymean` (Optional[bool]): Apply a moving average or rolling mean. 
 - `legend_template` (Optional[str]): A string that specifies the format of the legend. 
 - `gradient` (Optional[LList[GradientPoint]]): A list of gradient points that specify the color gradient of the plot. 
 - `font_size` (Optional[FontSize]): The size of the line plot's font. Options include `small`, `medium`, `large`, `auto`, or `None`. 
 - `regression` (Optional[bool]): If `True`, a regression line is plotted on the scatter plot. 







---



## <kbd>class</kbd> `SoundCloud`
A block that renders a SoundCloud player. 



**Attributes:**
 
 - `html` (str): The HTML code to embed the SoundCloud player. 







---



## <kbd>class</kbd> `Spotify`
A block that renders a Spotify player. 



**Attributes:**
 
 - `spotify_id` (str): The Spotify ID of the track or playlist. 







---



## <kbd>class</kbd> `SummaryMetric`
A summary metric to display in a report. 



**Attributes:**
 
 - `name` (str): The name of the metric. 







---



## <kbd>class</kbd> `TableOfContents`
A block that contains a list of sections and subsections using H1, H2, and H3 HTML blocks specified in a report. 







---



## <kbd>class</kbd> `TextWithInlineComments`
A block of text with inline comments. 



**Attributes:**
 
 - `text` (str): The text of the block. 







---



## <kbd>class</kbd> `Twitter`
A block that displays a Twitter feed. 



**Attributes:**
 
 - `html` (str): The HTML code to display the Twitter feed. 







---



## <kbd>class</kbd> `UnorderedList`
A list of items in a bulleted list. 



**Attributes:**
 
 - `items` (LList[str]): A list of one or more `UnorderedListItem` objects. 







---



## <kbd>class</kbd> `UnorderedListItem`
A list item in an unordered list. 



**Attributes:**
 
 - `text` (str): The text of the list item. 







---



## <kbd>class</kbd> `Video`
A block that renders a video. 



**Attributes:**
 
 - `url` (str): The URL of the video. 







---



## <kbd>class</kbd> `WeaveBlockArtifact`
A block that shows an artifact logged to W&B. The query takes the form of 

```python
project('entity', 'project').artifact('artifact-name')
``` 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `entity` (str): The entity that owns or has the appropriate permissions to the project where the artifact is stored. 
 - `project` (str): The project where the artifact is stored. 
 - `artifact` (str): The name of the artifact to retrieve. 
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: The tab to display in the artifact panel. 







---



## <kbd>class</kbd> `WeaveBlockArtifactVersionedFile`
A block that shows a versioned file logged to a W&B artifact. The query takes the form of 

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `entity` (str): The entity that owns or has the appropriate permissions to the project where the artifact is stored. 
 - `project` (str): The project where the artifact is stored. 
 - `artifact` (str): The name of the artifact to retrieve. 
 - `version` (str): The version of the artifact to retrieve. 
 - `file` (str): The name of the file stored in the artifact to retrieve. 







---



## <kbd>class</kbd> `WeaveBlockSummaryTable`
A block that shows a W&B Table, pandas DataFrame, plot, or other value logged to W&B. The query takes the form of 

```python
project('entity', 'project').runs.summary['value']
``` 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `entity` (str): The entity that owns or has the appropriate permissions to the project where the values are logged. 
 - `project` (str): The project where the value is logged in. 
 - `table_name` (str): The name of the table, DataFrame, plot, or value. 







---



## <kbd>class</kbd> `WeavePanel`
An empty query panel that can be used to display custom content using queries. 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 







---



## <kbd>class</kbd> `WeavePanelArtifact`
A panel that shows an artifact logged to W&B. 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `artifact` (str): The name of the artifact to retrieve. 
 - `tab Literal["overview", "metadata", "usage", "files", "lineage"]`: The tab to display in the artifact panel. 







---



## <kbd>class</kbd> `WeavePanelArtifactVersionedFile`
A panel that shows a versioned file logged to a W&B artifact. 

```python
project('entity', 'project').artifactVersion('name', 'version').file('file-name')
``` 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `artifact` (str): The name of the artifact to retrieve. 
 - `version` (str): The version of the artifact to retrieve. 
 - `file` (str): The name of the file stored in the artifact to retrieve. 







---



## <kbd>class</kbd> `WeavePanelSummaryTable`
A panel that shows a W&B Table, pandas DataFrame, plot, or other value logged to W&B. The query takes the form of 

```python
runs.summary['value']
``` 

The term "Weave" in the API name does not refer to the W&B Weave toolkit used for tracking and evaluating LLM. 



**Attributes:**
 
 - `table_name` (str): The name of the table, DataFrame, plot, or value. 





