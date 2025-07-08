---
title: Table
---

{{< cta-button githubLink=https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L198-L1036 >}}

The Table class used to display and analyze tabular data.

Unlike traditional spreadsheets, Tables support numerous types of data:
scalar values, strings, numpy arrays, and most subclasses of `wandb.data_types.Media`.
This means you can embed `Images`, `Video`, `Audio`, and other sorts of rich, annotated media
directly in Tables, alongside other traditional scalar values.

This class is the primary class used to generate the W&B Tables
https://docs.wandb.ai/guides/models/tables/.

## Methods

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L920-L961)

```python
add_column(
    name, data, optional=(False)
)
```

Adds a column of data to the table.

| Args |  |
| :--- | :--- |
|  `name` |  (str) - the unique name of the column |
|  `data` |  (list | np.array) - a column of homogeneous data |
|  `optional` |  (bool) - if null-like values are permitted |

### `add_computed_columns`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L1012-L1036)

```python
add_computed_columns(
    fn
)
```

Adds one or more computed columns based on existing data.

| Args |  |
| :--- | :--- |
|  `fn` |  A function which accepts one or two parameters, ndx (int) and row (dict), which is expected to return a dict representing new columns for that row, keyed by the new column names. `ndx` is an integer representing the index of the row. Only included if `include_ndx` is set to `True`. `row` is a dictionary keyed by existing columns |

### `add_data`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L506-L539)

```python
add_data(
    *data
)
```

Adds a new row of data to the table. The maximum amount of rows in a table is determined by `wandb.Table.MAX_ARTIFACT_ROWS`.

The length of the data should match the length of the table column.

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L500-L504)

```python
add_row(
    *row
)
```

Deprecated; use add_data instead.

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L409-L460)

```python
cast(
    col_name, dtype, optional=(False)
)
```

Casts a column to a specific data type.

This can be one of the normal python classes, an internal W&B type, or an
example object, like an instance of wandb.Image or wandb.Classes.

| Args |  |
| :--- | :--- |
|  `col_name` |  (str) - The name of the column to cast. |
|  `dtype` |  (class, wandb.wandb_sdk.interface._dtypes.Type, any) - The target dtype. |
|  `optional` |  (bool) - If the column should allow Nones. |

### `get_column`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L963-L986)

```python
get_column(
    name, convert_to=None
)
```

Retrieves a column from the table and optionally converts it to a NumPy object.

| Args |  |
| :--- | :--- |
|  `name` |  (str) - the name of the column |
|  `convert_to` |  (str, optional) - "numpy": will convert the underlying data to numpy object |

### `get_dataframe`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L997-L1003)

```python
get_dataframe()
```

Returns a `pandas.DataFrame` of the table.

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L988-L995)

```python
get_index()
```

Returns an array of row indexes for use in other tables to create links.

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L1005-L1010)

```python
index_ref(
    index
)
```

Gets a reference of the index of a row in the table.

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L797-L811)

```python
iterrows()
```

Returns the table data by row, showing the index of the row and the relevant data.

| Yields |  |
| :--- | :--- |

***

index : int
The index of the row. Using this value in other W&B tables
will automatically build a relationship between the tables
row : List[any]
The data of the row.

### `set_fk`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L819-L824)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/f1e324a66f6d9fd4ab7b43b66d9e832fa5e49b15/wandb/sdk/data_types/table.py#L813-L817)

```python
set_pk(
    col_name
)
```

| Class Variables |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |
