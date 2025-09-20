---
description: Learn how to build queries for custom charts and access different data tables
menu:
  default:
    identifier: query-syntax
    parent: custom-charts
title: Query syntax for custom charts
weight: 10
---

Learn how to build queries for custom charts to access different data tables in W&B. This guide covers both UI-based custom charts and programmatic chart creation using the Reports API.

## Understanding data tables

W&B provides different data tables you can query in custom charts:

### Summary data
Access final values from your runs using `summary`:
- Contains the last logged value for each metric
- Useful for comparing final results across runs
- Example: final accuracy, best validation score

### History data  
Access time-series data using `history`:
- Contains all logged values over time
- Useful for plotting training curves
- Example: loss over epochs, metrics over steps

### Summary tables
Access custom tables logged to summary using `summaryTable`:
- Contains `wandb.Table()` objects logged once per run
- Useful for structured data like evaluation results
- Requires specifying `tableKey` and `tableColumns`

### History tables
Access custom tables logged during training using `historyTable`:
- Contains `wandb.Table()` objects logged at multiple steps
- Each `wandb.log()` with a table creates a new entry
- Useful for progressive results like per-epoch predictions

## Query syntax examples

### Accessing summary metrics

To query summary metrics:

```json
{
  "summary": ["val_loss", "val_acc", "train_loss"]
}
```

### Accessing history metrics

To query time-series data:

```json
{
  "history": ["loss", "accuracy", "learning_rate"]
}
```

### Accessing summary tables

To query a custom table logged to summary:

```json
{
  "summaryTable": {
    "tableKey": "evaluation_results",
    "tableColumns": ["precision", "recall", "f1_score"]
  }
}
```

### Accessing history tables

To query tables logged at different time steps:

```json
{
  "historyTable": {
    "tableKey": "predictions_per_epoch",
    "tableColumns": ["epoch", "image", "prediction", "ground_truth"]
  }
}
```

## Using CustomChart with the reports API

When creating reports programmatically, use `wr.CustomChart` to add custom visualizations that query your logged data.

### Basic CustomChart examples

```python
import wandb_workspaces.reports.v2 as wr

# Create a chart from summary data
custom_chart = wr.CustomChart(
    query={'summary': ['val_loss', 'val_acc']},
    chart_name='wandb/scatter/v0',
    chart_fields={'x': 'val_loss', 'y': 'val_acc'}
)

# Add to a report
report = wr.Report(
    project="my-project",
    title="Custom Charts Report"
)

panel_grid = wr.PanelGrid(
    panels=[custom_chart],
    runsets=[wr.RunSet(entity="my-entity", project="my-project")]
)

report.blocks = [panel_grid]
report.save()
```

### Querying summary tables

To visualize data from a logged `wandb.Table`:

```python
# First, log a table in your training script
wandb.init()
eval_table = wandb.Table(
    columns=["class", "precision", "recall"],
    data=[
        ["cat", 0.95, 0.92],
        ["dog", 0.89, 0.94]
    ]
)
wandb.log({"eval_results": eval_table})

# Then, create a custom chart in your report
custom_chart = wr.CustomChart(
    query={
        'summaryTable': {
            'tableKey': 'eval_results',
            'tableColumns': ['class', 'precision', 'recall']
        }
    },
    chart_name='wandb/bar/v0',
    chart_fields={'x': 'class', 'y': 'precision'}
)
```

### Querying history tables

For tables logged at different time steps:

```python
# In your training script
for epoch in range(num_epochs):
    predictions_table = wandb.Table(
        columns=["x", "y", "prediction"],
        data=[[x, y, pred] for x, y, pred in epoch_predictions]
    )
    wandb.log({"predictions": predictions_table})

# In your report
custom_chart = wr.CustomChart(
    query={
        'historyTable': {
            'tableKey': 'predictions',
            'tableColumns': ['x', 'y', 'prediction']
        }
    },
    chart_name='wandb/scatter/v0',
    chart_fields={'x': 'x', 'y': 'y', 'color': 'prediction'}
)
```

### Combining data sources

You can query different tables in a single chart:

```python
custom_chart = wr.CustomChart(
    query={
        'summary': ['final_score'],
        'summaryTable': {
            'tableKey': 'class_scores',
            'tableColumns': ['class', 'score']
        }
    },
    chart_name='wandb/custom/v0',
    chart_fields={
        'x': 'class',
        'y': 'score',
        'title': 'Class Performance'
    }
)
```

## Importing existing charts to reports

To programmatically add charts that you've already created in the UI to a report:

### Method 1: Import charts via UI then access programmatically

```python
# Step 1: Create a report with imported charts in the UI
# Use "Create Report > Import charts" to add existing charts

# Step 2: Load the report and extract panels
import wandb_workspaces.reports.v2 as wr

# Load the report containing imported charts
source_report = wr.Report.from_url("https://wandb.ai/entity/project/reports/...")
imported_panels = source_report.blocks[0].panels

# Step 3: Add specific panels to a new report
target_report = wr.Report(
    project="my-project",
    title="Report with Imported Charts"
)

# Select the panel you want (by index or by filtering)
selected_panel = imported_panels[0]  # First panel

# Add to new report
panel_grid = wr.PanelGrid(
    panels=[selected_panel],
    runsets=[wr.RunSet(entity="my-entity", project="my-project")]
)

target_report.blocks = [panel_grid]
target_report.save()
```

### Method 2: Clone and update existing panels

```python
# Load a report as a template
template_report = wr.Report.from_url("https://wandb.ai/entity/project/reports/...")

# Extract and modify panels
for block in template_report.blocks:
    if isinstance(block, wr.PanelGrid):
        # Access existing panels
        existing_panels = block.panels
        
        # Modify or filter panels as needed
        modified_panels = []
        for panel in existing_panels:
            if isinstance(panel, wr.CustomChart):
                # You can access and modify the query
                panel.query['summaryTable']['tableKey'] = 'new_table_name'
            modified_panels.append(panel)
        
        # Create new panel grid with modified panels
        new_panel_grid = wr.PanelGrid(
            panels=modified_panels,
            runsets=block.runsets
        )

# Save the modified report
new_report = wr.Report(
    project="my-project", 
    title="Modified Report"
)
new_report.blocks = [new_panel_grid]
new_report.save()
```

## Common query patterns

### Filtering and grouping

When building queries, you can filter runs and group data:

```python
# Filter runs by config values
custom_chart = wr.CustomChart(
    query={
        'summary': ['accuracy'],
        'config': ['model_type', 'learning_rate']
    },
    chart_name='wandb/scatter/v0',
    chart_fields={
        'x': 'learning_rate',
        'y': 'accuracy',
        'color': 'model_type'
    }
)
```

### Multi-axis charts

Create charts with two y-axes:

```python
custom_chart = wr.CustomChart(
    query={'history': ['train_loss', 'val_loss', 'learning_rate']},
    chart_name='wandb/line/v0',
    chart_fields={
        'x': '_step',
        'y': ['train_loss', 'val_loss'],
        'y2': 'learning_rate'  # Secondary y-axis
    }
)
```

## Best practices

1. **Use appropriate table types**: 
   - Use `summaryTable` for final results that don't change
   - Use `historyTable` for data that evolves during training

2. **Limit table size**: Keep tables under 10,000 rows for optimal performance

3. **Name keys consistently**: Use descriptive, consistent names for table keys across your project

4. **Test queries in UI first**: Build and test your query in the UI custom chart editor, then transfer to code

5. **Cache imported panels**: When importing charts, load the source report once and reuse the panels

## Troubleshooting

### Data not appearing

If your data doesn't appear in the custom chart:
- Verify the table key matches exactly what you logged
- Check that selected runs contain the logged tables
- Ensure column names in `tableColumns` match your table's column names

### Query errors

Common query issues and solutions:
- **Missing quotes**: Ensure all strings in the query are properly quoted
- **Wrong field type**: `summaryTable` requires an object with `tableKey` and `tableColumns`, not an array
- **Typos in field names**: Double-check spelling of table keys and column names

## See also

- [Custom charts walkthrough]({{< relref "/guides/models/app/features/custom-charts/walkthrough.md" >}})
- [Reports API reference]({{< relref "/ref/python/wandb_workspaces/reports.md" >}})
- [Logging plots and media]({{< relref "/guides/models/track/log/plots.md" >}})
