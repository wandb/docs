# Table

<p><button style={{display: 'flex', alignItems: 'center', backgroundColor: 'white', border: '1px solid #ddd', padding: '10px', borderRadius: '6px', cursor: 'pointer', boxShadow: '0 2px 3px rgba(0,0,0,0.1)', transition: 'all 0.3s'}}><a href='https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L183-L909' style={{fontSize: '1.2em', display: 'flex', alignItems: 'center'}}><img src='https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png' height='32px' width='32px' style={{marginRight: '10px'}}/>View source on GitHub</a></button></p>


The Table class used to display and analyze tabular data.

```python
Table(
    columns=None, data=None, rows=None, dataframe=None, dtype=None, optional=(True),
    allow_mixed_types=(False)
)
```

Unlike traditional spreadsheets, Tables support numerous types of data:
scalar values, strings, numpy arrays, and most subclasses of `wandb.data_types.Media`.
This means you can embed `Images`, `Video`, `Audio`, and other sorts of rich, annotated media
directly in Tables, alongside other traditional scalar values.

This class is the primary class used to generate the Table Visualizer
in the UI: https://docs.wandb.ai/guides/data-vis/tables.

| Args |  |
| :--- | :--- |
|  `columns` |  (List[str]) Names of the columns in the table. Defaults to ["Input", "Output", "Expected"]. |
|  `data` |  (List[List[any]]) 2D row-oriented array of values. |
|  `dataframe` |  (pandas.DataFrame) DataFrame object used to create the table. When set, `data` and `columns` arguments are ignored. |
|  `optional` |  (Union[bool,List[bool]]) Determines if `None` values are allowed. Default to True - If a singular bool value, then the optionality is enforced for all columns specified at construction time - If a list of bool values, then the optionality is applied to each column - should be the same length as `columns` applies to all columns. A list of bool values applies to each respective column. |
|  `allow_mixed_types` |  (bool) Determines if columns are allowed to have mixed types (disables type validation). Defaults to False |

## Methods

### `add_column`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L797-L836)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L887-L909)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L423-L456)

```python
add_data(
    *data
)
```

Adds a new row of data to the table. The maximum amount of rows in a table is determined by `wandb.Table.MAX_ARTIFACT_ROWS`.

The length of the data should match the length of the table column.

### `add_row`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L418-L421)

```python
add_row(
    *row
)
```

Deprecated; use add_data instead.

### `cast`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L315-L371)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L838-L861)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L872-L878)

```python
get_dataframe()
```

Returns a `pandas.DataFrame` of the table.

### `get_index`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L863-L870)

```python
get_index()
```

Returns an array of row indexes for use in other tables to create links.

### `index_ref`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L880-L885)

```python
index_ref(
    index
)
```

Gets a reference of the index of a row in the table.

### `iterrows`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L674-L688)

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

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L695-L699)

```python
set_fk(
    col_name, table, table_col
)
```

### `set_pk`

[View source](https://www.github.com/wandb/wandb/tree/v0.18.6/wandb/sdk/data_types/table.py#L690-L693)

```python
set_pk(
    col_name
)
```

| Class Variables |  |
| :--- | :--- |
|  `MAX_ARTIFACT_ROWS`<a id="MAX_ARTIFACT_ROWS"></a> |  `200000` |
|  `MAX_ROWS`<a id="MAX_ROWS"></a> |  `10000` |
