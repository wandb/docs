---
title: Custom Charts
module: wandb.plot
weight: 5
no_list: true
---

Custom Charts in W&B (`wandb.plot`) are visualization functions that create interactive visualizations that integrate with the W&B UI. These functions handle common ML visualization requirements such as confusion matrices, ROC curves, and distribution plots. Custom charts can also be created using Vega-Lite specifications.

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
- **Vega-Lite Charts**: `plot_table()` for domain-specific visualizations

### Training Monitoring
- **Learning Curves**: `line()` or `line_series()` for tracking metrics over epochs
- **Hyperparameter Comparison**: `bar()` charts for comparing configurations

### Data Analysis
- **Distribution Analysis**: `histogram()` for feature distributions
- **Correlation Analysis**: `scatter()` plots for variable relationships

## Getting Started

### Log a confusion matrix

```python
import wandb

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 2, 0, 1, 1]
class_names = ["class_0", "class_1", "class_2"]

# Initialize a run
with wandb.init(project="custom-charts-demo") as run:
    run.log({
        "conf_mat": wandb.plot.confusion_matrix(
            y_true=y_true, 
            preds=y_pred,
            class_names=class_names
        )
    })
```


### Build a scatter plot for feature analysis
```python
import numpy as np

# Generate synthetic data
data_table = wandb.Table(columns=["feature_1", "feature_2", "label"])

with wandb.init(project="custom-charts-demo") as run:

    for _ in range(100):
        data_table.add_data(
            np.random.randn(), 
            np.random.randn(), 
            np.random.choice(["A", "B"])
        )

    run.log({
        "feature_scatter": wandb.plot.scatter(
            data_table, x="feature_1", y="feature_2",
            title="Feature Distribution"
        )
    })

```
