---
title: pr_curve()
object_type: python_sdk_plot_namespace
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/pr_curve.py >}}




### <kbd>function</kbd> `pr_curve`

```python
pr_curve(
    y_true: 'Iterable[T] | None' = None,
    y_probas: 'Iterable[numbers.Number] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[T] | None' = None,
    interp_size: 'int' = 21,
    title: 'str' = 'Precision-Recall Curve',
    split_table: 'bool' = False
) → CustomChart
```

Constructs a Precision-Recall (PR) curve. 

The Precision-Recall curve is particularly useful for evaluating classifiers on imbalanced datasets. A high area under the PR curve signifies both high precision (a low false positive rate) and high recall (a low false negative rate). The curve provides insights into the balance between false positives and false negatives at various threshold levels, aiding in the assessment of a model's performance. 



**Args:**
 
 - `y_true`:  True binary labels. The shape should be (`num_samples`,). 
 - `y_probas`:  Predicted scores or probabilities for each class.  These can be probability estimates, confidence scores, or non-thresholded  decision values. The shape should be (`num_samples`, `num_classes`). 
 - `labels`:  Optional list of class names to replace  numeric values in `y_true` for easier plot interpretation.  For example, `labels = ['dog', 'cat', 'owl']` will replace 0 with  'dog', 1 with 'cat', and 2 with 'owl' in the plot. If not provided,  numeric values from `y_true` will be used. 
 - `classes_to_plot`:  Optional list of unique class values from  y_true to be included in the plot. If not specified, all unique  classes in y_true will be plotted. 
 - `interp_size`:  Number of points to interpolate recall values. The  recall values will be fixed to `interp_size` uniformly distributed  points in the range [0, 1], and the precision will be interpolated  accordingly. 
 - `title`:  Title of the plot. Defaults to "Precision-Recall Curve". 
 - `split_table`:  Whether the table should be split into a separate section  in the W&B UI. If `True`, the table will be displayed in a section named  "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 



**Raises:**
 
 - `wandb.Error`:  If NumPy, pandas, or scikit-learn is not installed. 





**Example:**
 

```python
import wandb

# Example for spam detection (binary classification)
y_true = [0, 1, 1, 0, 1]  # 0 = not spam, 1 = spam
y_probas = [
    [0.9, 0.1],  # Predicted probabilities for the first sample (not spam)
    [0.2, 0.8],  # Second sample (spam), and so on
    [0.1, 0.9],
    [0.8, 0.2],
    [0.3, 0.7],
]

labels = ["not spam", "spam"]  # Optional class names for readability

with wandb.init(project="spam-detection") as run:
    pr_curve = wandb.plot.pr_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=labels,
         title="Precision-Recall Curve for Spam Detection",
    )
    run.log({"pr-curve": pr_curve})
``` 
