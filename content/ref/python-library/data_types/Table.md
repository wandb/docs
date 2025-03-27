---
title: Table
object_type: data_type
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/sdk/data_types/table.py >}}




## <kbd>class</kbd> `Table`
The Table class used to display and analyze tabular data. 

Unlike traditional spreadsheets, Tables support numerous types of data: scalar values, strings, numpy arrays, and most subclasses of `wandb.data_types.Media`. This means you can embed `Images`, `Video`, `Audio`, and other sorts of rich, annotated media directly in Tables, alongside other traditional scalar values. 

This class is the primary class used to generate the Table Visualizer in the UI: https://docs.wandb.ai/guides/data-vis/tables. 



**Attributes:**
 
 - `columns` (List[str]):  Names of the columns in the table.  Defaults to ["Input", "Output", "Expected"]. 
 - `data`:  (List[List[any]]) 2D row-oriented array of values. 
 - `dataframe` (pandas.DataFrame):  DataFrame object used to create the table.  When set, `data` and `columns` arguments are ignored. 
 - `optional` (Union[bool,List[bool]]):  Determines if `None` values are  allowed. Default to `True`. 
        - If a singular bool value, then the optionality is enforced for all  columns specified at construction time. 
        - If a list of bool values, then the optionality is applied to each  column - should be the same length as `columns`.  applies to all columns. A list of bool values applies to each  respective column. 
 - `allow_mixed_types` (bool):  Determines if columns are allowed to have  mixed types (disables type validation). Defaults to False. 

### <kbd>method</kbd> `Table.__init__`

```python
__init__(
    columns=None,
    data=None,
    rows=None,
    dataframe=None,
    dtype=None,
    optional=True,
    allow_mixed_types=False
)
```

Initializes a Table object. 

The rows is available for legacy reasons and should not be used. The Table class uses data to mimic the Pandas API. 




---

### <kbd>method</kbd> `Table.add_column`

```python
add_column(name, data, optional=False)
```

Adds a column of data to the table. 



**Args:**
 
 - `name`:  (str) - the unique name of the column 
 - `data`:  (list | np.array) - a column of homogeneous data 
 - `optional`:  (bool) - if null-like values are permitted 

---

### <kbd>method</kbd> `Table.add_computed_columns`

```python
add_computed_columns(fn)
```

Adds one or more computed columns based on existing data. 



**Args:**
 
 - `fn`:  A function which accepts one or two parameters, ndx (int) and row (dict),  which is expected to return a dict representing new columns for that row, keyed  by the new column names. 

 `ndx` is an integer representing the index of the row. Only included if `include_ndx`  is set to `True`. 

 `row` is a dictionary keyed by existing columns 

---

### <kbd>method</kbd> `Table.add_data`

```python
add_data(*data)
```

Adds a new row of data to the table. 

The maximum amount ofrows in a table is determined by `wandb.Table.MAX_ARTIFACT_ROWS`. 

The length of the data should match the length of the table column. 

---

### <kbd>method</kbd> `Table.add_row`

```python
add_row(*row)
```

Deprecated; use add_data instead. 

---


### <kbd>method</kbd> `Table.cast`

```python
cast(col_name, dtype, optional=False)
```

Casts a column to a specific data type. 

This can be one of the normal python classes, an internal W&B type, or an example object, like an instance of wandb.Image or wandb.Classes. 



**Args:**
 
 - `col_name` (str):  The name of the column to cast. 
 - `dtype` (class, wandb.wandb_sdk.interface._dtypes.Type, any):  The  target dtype. 
 - `optional` (bool):  If the column should allow Nones. 

---


### <kbd>method</kbd> `Table.get_column`

```python
get_column(name, convert_to=None)
```

Retrieves a column from the table and optionally converts it to a NumPy object. 



**Args:**
 
 - `name`:  (str) - the name of the column 
 - `convert_to`:  (str, optional) 
        - "numpy": will convert the underlying data to numpy object 

---

### <kbd>method</kbd> `Table.get_dataframe`

```python
get_dataframe()
```

Returns a `pandas.DataFrame` of the table. 

---

### <kbd>method</kbd> `Table.get_index`

```python
get_index()
```

Returns an array of row indexes for use in other tables to create links. 

---






