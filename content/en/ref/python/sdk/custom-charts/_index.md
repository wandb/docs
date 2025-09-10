---
title: Custom Charts
module: wandb.plot
weight: 5
---

The W&B Python SDK includes functions for creating specialized charts from your data. These functions generate interactive visualizations that integrate with the W&B UI.

## Overview

Custom Charts in W&B (`wandb.plot`) are visualization functions that transform data into interactive charts. These functions handle common ML visualization requirements such as confusion matrices, ROC curves, and distribution plots. Custom charts can also be created using Vega-Lite specifications.

## Available Chart Functions

| Function | Description |
|----------|-------------|
| [`confusion_matrix()`](./confusion_matrix/) | Generate confusion matrices for classification performance visualization. |
| [`roc_curve()`](./roc_curve/) | Create Receiver Operating Characteristic curves for binary and multi-class classifiers. |
| [`pr_curve()`](./pr_curve/) | Build Precision-Recall curves for classifier evaluation. |
| [`line()`](./line/) | Construct line charts from tabular data. |
| [`scatter()`](./scatter/) | Create scatter plots for variable relationships. |
| [`bar()`](./bar/) | Generate bar charts for categorical data. |
| [`histogram()`](./Histogram/) | Build histograms for data distribution analysis. |
| [`line_series()`](./line_series/) | Plot multiple line series on a single chart. |
| [`plot_table()`](./plot_table/) | Create custom charts using Vega-Lite specifications. |

## Common Use Cases

### Model Evaluation
- **Classification**: `confusion_matrix()`, `roc_curve()`, and `pr_curve()` for classifier evaluation
- **Regression**: `scatter()` for prediction vs. actual plots and `histogram()` for residual analysis

### Training Monitoring
- **Learning Curves**: `line()` or `line_series()` for tracking metrics over epochs
- **Hyperparameter Comparison**: `bar()` charts for comparing configurations

### Data Analysis
- **Distribution Analysis**: `histogram()` for feature distributions
- **Correlation Analysis**: `scatter()` plots for variable relationships

### Custom Visualizations
- **Vega-Lite Charts**: `plot_table()` for domain-specific visualizations
- **Multi-Chart Dashboards**: Combination of multiple chart types in a single run

## Example: Scatter plot

```python
import wandb
import numpy as np
from sklearn.metrics import roc_curve as sklearn_roc_curve

# Initialize a run
wandb.init(project="custom-charts-demo")

# Example 1: Log a confusion matrix
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 2, 0, 1, 1]
class_names = ["class_0", "class_1", "class_2"]

wandb.log({
    "conf_mat": wandb.plot.confusion_matrix(
        y_true=y_true, 
        preds=y_pred,
        class_names=class_names
    )
})

# Example 2: Create an ROC curve for binary classification
y_true_binary = np.array([0, 0, 1, 1, 0, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])

wandb.log({
    "roc": wandb.plot.roc_curve(
        y_true=y_true_binary, 
        y_probas=y_scores,
        labels=["negative", "positive"]
    )
})

# Example 3: Create a custom line chart from a table
table = wandb.Table(columns=["epoch", "accuracy", "loss"])
for epoch in range(10):
    table.add_data(epoch, 0.8 + 0.02 * epoch, 1.0 - 0.1 * epoch)

wandb.log({
    "training_progress": wandb.plot.line(
        table, x="epoch", y="accuracy",
        title="Model Accuracy Over Time"
    )
})

# Example 4: Build a scatter plot for feature analysis
data_table = wandb.Table(columns=["feature_1", "feature_2", "label"])
for _ in range(100):
    data_table.add_data(
        np.random.randn(), 
        np.random.randn(), 
        np.random.choice(["A", "B"])
    )

wandb.log({
    "feature_scatter": wandb.plot.scatter(
        data_table, x="feature_1", y="feature_2",
        title="Feature Distribution"
    )
})

wandb.finish()
```
