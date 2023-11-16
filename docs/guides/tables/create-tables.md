---
description: How to create a table.
displayed_sidebar: default
---
# Create a table

Unlike traditional spreadsheets, Tables support numerous types of data:
scalar values, strings, numpy arrays, and most subclasses of `wandb.data_types.Media`.
This means you can embed `Images`, `Video`, `Audio`, and other sorts of rich, annotated media
directly in Tables, alongside other traditional scalar values.

Tables can be constructed with initial data using the `data` or
`dataframe` parameters:

<!--yeadoc-test:table-construct-dataframe-->


```python
import pandas as pd
import wandb

data = {"users": ["geoff", "juergen", "ada"], "feature_01": [1, 117, 42]}
df = pd.DataFrame(data)

tbl = wandb.Table(data=df)
assert all(tbl.get_column("users") == df["users"])
assert all(tbl.get_column("feature_01") == df["feature_01"])
```

Additionally, users can add data to Tables incrementally by using the
`add_data`, `add_column`, and `add_computed_column` functions for
adding rows, columns, and columns computed from data in other columns, respectively:

<!--yeadoc-test:table-construct-rowwise-->


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

Tables can be logged directly to runs using `run.log({"my_table": table})`
or added to artifacts using `artifact.add(table, "my_table")`:

<!--yeadoc-test:table-logging-direct-->


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

Tables added directly to runs as above will produce a corresponding Table Visualizer in the
Workspace which can be used for further analysis and exporting to reports.

Tables added to artifacts can be viewed in the Artifact Tab and will render
an equivalent Table Visualizer directly in the artifact browser.

Tables expect each value for a column to be of the same type. By default, a column supports
optional values, but not mixed values. If you absolutely need to mix types,
you can enable the `allow_mixed_types` flag which will disable type checking
on the data. This will result in some table analytics features being disabled
due to lack of consistent typing.