---
title: histogram_wandb()
namespace: python_sdk_custom_charts
python_object_type: function
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/plot/histogram.py >}}




### <kbd>function</kbd> `histogram`

```python
histogram(
    table: 'wandb.Table',
    value: 'str',
    title: 'str' = '',
    split_table: 'bool' = False
) → CustomChart
```

Constructs a histogram chart from a W&B Table. 



**Args:**
 
 - `table`:  The W&B Table containing the data for the histogram. 
 - `value`:  The label for the bin axis (x-axis). 
 - `title`:  The title of the histogram plot. 
 - `split_table`:  Whether the table should be split into a separate section  in the W&B UI. If `True`, the table will be displayed in a section named  "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 



**Example:**
 

```python
import math
import random
import wandb

# Generate random data
data = [[i, random.random() + math.sin(i / 10)] for i in range(100)]

# Create a W&B Table
table = wandb.Table(
    data=data,
    columns=["step", "height"],
)

# Create a histogram plot
histogram = wandb.plot.histogram(
    table,
    value="height",
    title="My Histogram",
)

# Log the histogram plot to W&B
with wandb.init(...) as run:
    run.log({"histogram-plot1": histogram})
``` 
