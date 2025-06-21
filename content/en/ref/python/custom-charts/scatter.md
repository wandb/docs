---
title: scatter()
object_type: python_sdk_custom_charts
data_type_classification: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/scatter.py >}}




### <kbd>function</kbd> `scatter`

```python
scatter(
    table: 'wandb.Table',
    x: 'str',
    y: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

Constructs a scatter plot from a wandb.Table of data. 



**Args:**
 
 - `table`:  The W&B Table containing the data to visualize. 
 - `x`:  The name of the column used for the x-axis. 
 - `y`:  The name of the column used for the y-axis. 
 - `title`:  The title of the scatter chart. 
 - `split_table`:  Whether the table should be split into a separate section  in the W&B UI. If `True`, the table will be displayed in a section named  "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 

**Example:**
 ```python
import math
import random
import wandb

# Simulate temperature variations at different altitudes over time
data = [
    [i, random.uniform(-10, 20) - 0.005 * i + 5 * math.sin(i / 50)]
    for i in range(300)
]

# Create W&B table with altitude (m) and temperature (°C) columns
table = wandb.Table(data=data, columns=["altitude (m)", "temperature (°C)"])

# Initialize W&B run and log the scatter plot
with wandb.init(project="temperature-altitude-scatter") as run:
    # Create and log the scatter plot
    scatter_plot = wandb.plot.scatter(
         table=table,
         x="altitude (m)",
         y="temperature (°C)",
         title="Altitude vs Temperature",
    )
    run.log({"altitude-temperature-scatter": scatter_plot})
``` 
