---
description: Create tables with W&B.
displayed_sidebar: default
---

# Create a table

Unlike traditional spreadsheets, Tables support numerous types of data. This includes
scalar values, strings, NumPy arrays, and most subclasses of `wandb.data_types.Media`.
You can embed `Images`, `Video`, `Audio`, and other sorts of rich, annotated media
directly in Tables, alongside other traditional scalar values.

Construct tables with initial data by creating a dataframe:


```python
import pandas as pd
import wandb

#Intialize the data
data = {"users": ["geoff", "juergen", "ada"], "feature_01": [1, 117, 42]}
#Turn the data into a dataframe.
df = pd.DataFrame(data)

tbl = wandb.Table(data=df)
assert all(tbl.get_column("users") == df["users"])
assert all(tbl.get_column("feature_01") == df["feature_01"])
```
## Add data

Add data to Tables incrementally by using the
`add_data`, `add_column`, and `add_computed_column` functions for
adding rows, columns, and columns computed from data in other columns, respectively:


```python
import wandb

tbl = wandb.Table(columns=["user"])

users = ["geoff", "juergen", "ada"]

[tbl.add_data(user) for user in users]
assert tbl.get_column("user") == users


def get_user_name_length(index, row):
    return {"feature_01": len(row["user"])}


tbl.add_computed_columns(get_user_name_length)
assert tbl.get_column("feature_01") == [5, 7, 3]
```

## Log data

Log tables directly to runs with `run.log({"my_table": table})`
or add them to artifacts using `artifact.add(table, "my_table")`:


```python
import numpy as np
import wandb

wandb.init()

tbl = wandb.Table(columns=["image", "label"])

images = np.random.randint(0, 255, [2, 100, 100, 3], dtype=np.uint8)
labels = ["panda", "gibbon"]
[tbl.add_data(wandb.Image(image), label) for image, label in zip(images, labels)]

wandb.log({"classifier_out": tbl})
```

Tables added directly to runs produces a corresponding Table Visualizer in the
Workspace which can further analyze data and export it to reports.

Tables added to artifacts appear in the Artifact tab and render
an equivalent Table Visualizer directly in the artifact browser.

## Best practices
Tables expect each value for a column to be of the same type. By default, a column supports optional values, but not mixed values. If you absolutely need to mix types, you can enable the `allow_mixed_types` flag which turns off type checking on the data. This results in some table analytics features turning off due to lack of consistent typing.

## Next steps
- For more information about visualizing tables, see the [table visualization guide](./visualize-tables.md).
- For a more in-depth walkthrough of how to use tables, see the [walkthrough](tables-walkthrough.md).