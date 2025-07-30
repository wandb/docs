---
description: Some features on this page are in beta, hidden behind a feature flag.
  Add `weave-plot` to your bio on your profile page to unlock all related features.
url: guides/app/features/panels/query-panels
menu:
  default:
    identifier: intro_query_panel
    parent: panels
cascade:
- url: guides/app/features/panels/query-panels/:filename
title: Query panels
---


{{% alert %}}
Looking for W&B Weave? W&B's suite of tools for Generative AI application building? Find the docs for weave here: [wandb.me/weave](https://wandb.github.io/weave/?utm_source=wandb_docs&utm_medium=docs&utm_campaign=weave-nudge).
{{% /alert %}}

Use query panels to query and interactively visualize your data.

{{< img src="/images/weave/pretty_panel.png" alt="Query panel" >}}

<!-- {{% alert %}}
See the [Keras XLA benchmark report](http://wandb.me/keras-xla-benchmark) to see how this team used query panels to visualize their benchmarks.
{{% /alert %}} -->

## Create a query panel

Add a query to your workspace or within a report.

{{< tabpane text=true >}}
{{% tab header="Project workspace" value="workspace" %}}

  1. Navigate to your project's workspace. 
  2. In the upper right hand corner, click `Add panel`.
  3. From the dropdown, select `Query panel`.
  {{< img src="/images/weave/add_weave_panel_workspace.png" alt="Add panel dropdown" >}}

{{% /tab %}}

{{% tab header="W&B Report" value="report" %}}

Type and select `/Query panel`.

{{< img src="/images/weave/add_weave_panel_report_1.png" alt="Query panel option" >}}

Alternatively, you can associate a query with a set of runs:
1. Within your report, type and select `/Panel grid`.
2. Click the `Add panel` button.
3. From the dropdown, select `Query panel`.

{{% /tab %}}
{{< /tabpane >}}
  

## Query components

### Expressions

Use query expressions to query your data stored in W&B such as runs, artifacts, models, tables, and more. 

#### Example: Query a table
Suppose you want to query a W&B Table. In your training code you log a table called `"cifar10_sample_table"`:

```python
import wandb
with wandb.init() as run:
  run.log({"cifar10_sample_table":<MY_TABLE>})
```

Within the query panel you can query your table with:
```python
runs.summary["cifar10_sample_table"]
```
{{< img src="/images/weave/basic_weave_expression.png" alt="Table query expression" >}}

Breaking this down:

* `runs` is a variable automatically injected in Query Panel Expressions when the Query Panel is in a Workspace. Its "value" is the list of runs which are visible for that particular Workspace. [Read about the different attributes available within a run here]({{< relref "../../../../track/public-api-guide.md#understanding-the-different-attributes" >}}).
* `summary` is an op which returns the Summary object for a Run. Ops are _mapped_, meaning this op is applied to each Run in the list, resulting in a list of Summary objects.
* `["cifar10_sample_table"]` is a Pick op (denoted with brackets), with a parameter of `predictions`. Since Summary objects act like dictionaries or maps, this operation picks the `predictions` field off of each Summary object.

To learn how to write your own queries interactively, see the [Query panel demo](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr).

### Configurations

Select the gear icon on the upper left corner of the panel to expand the query configuration. This allows the user to configure the type of panel and the parameters for the result panel.

{{< img src="/images/weave/weave_panel_config.png" alt="Panel configuration menu" >}}

### Result panels

Finally, the query result panel renders the result of the query expression, using the selected query panel, configured by the configuration to display the data in an interactive form. The following images shows a Table and a Plot of the same data.

{{< img src="/images/weave/result_panel_table.png" alt="Table result panel" >}}

{{< img src="/images/weave/result_panel_plot.png" alt="Plot result panel" >}}

## Basic operations
The following common operations you can make within your query panels.

### Available query operations reference

| Operation | Description | Example |
|-----------|-------------|---------|
| `.concat` | Concatenate multiple arrays/tables into one | `table1.rows.concat` |
| `.groupby(fn)` | Group data by a function result | `.groupby((row) => row["category"])` |
| `.map(fn)` | Transform each element | `.map((item) => item * 2)` |
| `.filter(fn)` | Filter elements by condition | `.filter((row) => row["value"] > 10)` |
| `.sort(fn)` | Sort elements | `.sort((a, b) => a["timestamp"] - b["timestamp"])` |
| `.groupkey` | Access the key of a group (after groupby) | `group.groupkey` |
| `.avg` | Calculate average of numeric values | `group["metric"].avg` |
| `.min` | Find minimum value | `group["metric"].min` |
| `.max` | Find maximum value | `group["metric"].max` |
| `.sum` | Sum all values | `group["metric"].sum` |
| `.count` | Count number of items | `group["metric"].count` |
| `.join()` | Join two tables | See Join section below |
### Sort
Sort from the column options:
{{< img src="/images/weave/weave_sort.png" alt="Column sort options" >}}

### Filter
You can either filter directly in the query or using the filter button in the top left corner (second image)
{{< img src="/images/weave/weave_filter_1.png" alt="Query filter syntax" >}}
{{< img src="/images/weave/weave_filter_2.png" alt="Filter button" >}}

### Map
Map operations iterate over lists and apply a function to each element in the data. You can do this directly with a panel query  or by inserting a new column from the column options.
{{< img src="/images/weave/weave_map.png" alt="Map operation query" >}}
{{< img src="/images/weave/weave_map.gif" alt="Map column insertion" >}}

### Groupby
You can groupby using a query or from the column options.
{{< img src="/images/weave/weave_groupby.png" alt="Group by query" >}}
{{< img src="/images/weave/weave_groupby.gif" alt="Group by column options" >}}

#### Groupby with aggregations
After grouping data, you can apply aggregation functions to compute metrics for each group. Common aggregation operations include:

- `.avg` - Calculate the average of numeric values
- `.min` - Find the minimum value
- `.max` - Find the maximum value  
- `.sum` - Sum all values
- `.count` - Count the number of items

##### Example: Group runs by model and calculate average metrics

```javascript
// Group runs by model type and calculate average loss and accuracy
runs.summary["results_table"].table.rows.concat
  .groupby((row) => row["model"])
  .map((row, index) => ({
    model: row.groupkey,
    loss_mean: row["loss"].avg,
    accuracy_mean: row["accuracy"].avg,
    run_count: row["loss"].count
  }))
```

This example:
1. Concatenates all rows from the `results_table` 
2. Groups rows by the `model` field
3. For each group, creates a new object with:
   - `model`: The group key (using `.groupkey`)
   - `loss_mean`: Average loss for that model
   - `accuracy_mean`: Average accuracy for that model
   - `run_count`: Number of runs for that model

#### Example: Working with summary tables (customer use case)
If you've logged a summary table with multiple metrics per run, you can aggregate across all runs:

```javascript
// Original customer example - group by model and calculate averages
runs.summary["summary_table"].table.rows.concat
  .groupby((row) => row["model"])
  .map((row, index) => ({
    model: row.groupkey,
    loss_mean: row["loss"].avg,
    accuracy_mean: row["accuracy"].avg
  }))
```

This pattern is useful when you've logged evaluation results as a table and want to compute aggregate statistics across different model configurations or experiments.

##### Example: Find best performing configuration

```javascript
// Group by hyperparameter configuration and find best accuracy
runs.config.groupby((config) => config["optimizer"])
  .map((group) => ({
    optimizer: group.groupkey,
    best_accuracy: group.summary["accuracy"].max,
    worst_accuracy: group.summary["accuracy"].min,
    avg_accuracy: group.summary["accuracy"].avg
  }))
```

### Advanced query examples

#### Example: Aggregate metrics across multiple runs
```javascript
// Get average, min, and max accuracy across all runs
runs.map((run) => ({
  name: run.name,
  accuracy: run.summary["accuracy"],
  loss: run.summary["loss"]
}))
.concat
.map((allRuns) => ({
  avg_accuracy: allRuns["accuracy"].avg,
  min_accuracy: allRuns["accuracy"].min,
  max_accuracy: allRuns["accuracy"].max,
  avg_loss: allRuns["loss"].avg
}))
```

#### Example: Compare model performance by dataset
```javascript
// Group runs by dataset and model, then compare performance
runs.summary["eval_results"].table.rows.concat
  .groupby((row) => row["dataset"] + "_" + row["model"])
  .map((group) => {
    const [dataset, model] = group.groupkey.split("_");
    return {
      dataset: dataset,
      model: model,
      avg_f1_score: group["f1_score"].avg,
      avg_precision: group["precision"].avg,
      avg_recall: group["recall"].avg,
      sample_count: group["f1_score"].count
    };
  })
  .sort((a, b) => b["avg_f1_score"] - a["avg_f1_score"])
```

#### Example: Calculate custom metrics from logged data
```javascript
// Calculate error rate and other derived metrics
runs.history()
  .map((history) => ({
    step: history["_step"],
    error_rate: 1 - history["accuracy"],
    loss_accuracy_ratio: history["loss"] / history["accuracy"],
    is_improving: history["accuracy"] > 0.9
  }))
```

### Concat
The concat operation allows you to concatenate 2 tables and concatenate or join from the panel settings
{{< img src="/images/weave/weave_concat.gif" alt="Table concatenation" >}}

### Join
It is also possible to join tables directly in the query. Consider the following query expression:
```python
project("luis_team_test", "weave_example_queries").runs.summary["short_table_0"].table.rows.concat.join(\
project("luis_team_test", "weave_example_queries").runs.summary["short_table_1"].table.rows.concat,\
(row) => row["Label"],(row) => row["Label"], "Table1", "Table2",\
"false", "false")
```
{{< img src="/images/weave/weave_join.png" alt="Table join operation" >}}

The table on the left is generated from:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_0"].table.rows.concat.join
```
The table in the right is generated from:
```python
project("luis_team_test", "weave_example_queries").\
runs.summary["short_table_1"].table.rows.concat
```
Where:
* `(row) => row["Label"]` are selectors for each table, determining which column to join on
* `"Table1"` and `"Table2"` are the names of each table when joined
* `true` and `false` are for left and right inner/outer join settings


## Runs object
Use query panels to access the `runs` object. Run objects store records of your experiments. You can find more details in [Accessing runs object](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#3.-accessing-runs-object) but, as quick overview, `runs` object has available:
* `summary`: A dictionary of information that summarizes the run's results. This can be scalars like accuracy and loss, or large files. By default, `wandb.Run.log()` sets the summary to the final value of a logged time series. You can set the contents of the summary directly. Think of the summary as the run's outputs.
* `history`: A list of dictionaries meant to store values that change while the model is training such as loss. The command `wandb.Run.log()` appends to this object.
* `config`: A dictionary of the run's configuration information, such as the hyperparameters for a training run or the preprocessing methods for a run that creates a dataset Artifact. Think of these as the run's "inputs"
{{< img src="/images/weave/weave_runs_object.png" alt="Runs object structure" >}}

## Troubleshooting

### Autocomplete not showing all operations
If autocomplete doesn't show certain operations like `.groupkey` or `.avg`, you can still use them - they are valid operations. Common operations that might not appear in autocomplete but are available:

- After `groupby()`: `.groupkey` to access the group identifier
- On numeric arrays: `.avg`, `.min`, `.max`, `.sum`, `.count`
- On tables: `.rows`, `.concat`
- On runs: `.summary`, `.history()`, `.config`

### Type conversion
Some operations require specific data types:
- Aggregation functions (`.avg`, `.sum`) only work on numeric values
- String operations won't work on numbers without conversion
- Use `.map()` to convert types if needed

## Access Artifacts

Artifacts are a core concept in W&B. They are a versioned, named collection of files and directories. Use Artifacts to track model weights, datasets, and any other file or directory. Artifacts are stored in W&B and can be downloaded or used in other runs. You can find more details and examples in [Accessing artifacts](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr#4.-accessing-artifacts). Artifacts are normally accessed from the `project` object:
* `project.artifactVersion()`: returns the specific artifact version for a given name and version within a project
* `project.artifact("")`: returns the artifact for a given name within a project. You can then use `.versions` to get a list of all versions of this artifact
* `project.artifactType()`: returns the `artifactType` for a given name within a project. You can then use `.artifacts` to get a list of all artifacts with this type
* `project.artifactTypes`: returns a list of all artifact types under the project
{{< img src="/images/weave/weave_artifacts.png" alt="Artifact access methods" >}}