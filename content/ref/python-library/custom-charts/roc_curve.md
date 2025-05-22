---
title: roc_curve()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/plot/roc_curve.py >}}




### <kbd>function</kbd> `roc_curve`

```python
roc_curve(
    y_true: 'Sequence[numbers.Number]',
    y_probas: 'Sequence[Sequence[float]] | None' = None,
    labels: 'list[str] | None' = None,
    classes_to_plot: 'list[numbers.Number] | None' = None,
    title: 'str' = 'ROC Curve',
    split_table: 'bool' = False
) → CustomChart
```

Constructs Receiver Operating Characteristic (ROC) curve chart. 



**Args:**
 
 - `y_true`:  The true class labels (ground truth)  for the target variable. Shape should be (num_samples,). 
 - `y_probas`:  The predicted probabilities or  decision scores for each class. Shape should be (num_samples, num_classes). 
 - `labels`:  Human-readable labels corresponding to the class  indices in `y_true`. For example, if `labels=['dog', 'cat']`,  class 0 will be displayed as 'dog' and class 1 as 'cat' in the plot.  If None, the raw class indices from `y_true` will be used.  Default is None. 
 - `classes_to_plot`:  A subset of unique class labels  to include in the ROC curve. If None, all classes in `y_true` will  be plotted. Default is None. 
 - `title`:  Title of the ROC curve plot. Default is "ROC Curve". 
 - `split_table`:  Whether the table should be split into a separate  section in the W&B UI. If `True`, the table will be displayed in a  section named "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 



**Raises:**
 
 - `wandb.Error`:  If numpy, pandas, or scikit-learn are not found. 



**Example:**
 ```python
import numpy as np
import wandb

# Simulate a medical diagnosis classification problem with three diseases
n_samples = 200
n_classes = 3

# True labels: assign "Diabetes", "Hypertension", or "Heart Disease" to
# each sample
disease_labels = ["Diabetes", "Hypertension", "Heart Disease"]
# 0: Diabetes, 1: Hypertension, 2: Heart Disease
y_true = np.random.choice([0, 1, 2], size=n_samples)

# Predicted probabilities: simulate predictions, ensuring they sum to 1
# for each sample
y_probas = np.random.dirichlet(np.ones(n_classes), size=n_samples)

# Specify classes to plot (plotting all three diseases)
classes_to_plot = [0, 1, 2]

# Initialize a W&B run and log a ROC curve plot for disease classification
with wandb.init(project="medical_diagnosis") as run:
    roc_plot = wandb.plot.roc_curve(
         y_true=y_true,
         y_probas=y_probas,
         labels=disease_labels,
         classes_to_plot=classes_to_plot,
         title="ROC Curve for Disease Classification",
    )
    run.log({"roc-curve": roc_plot})
``` 
