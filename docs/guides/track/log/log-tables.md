# Log Tables

The simplest way to log a table is to log a [pandas dataframe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) which will be automatically converted into a W&B Table.

```python
wandb.log({"table": my_dataframe})
```

![Tables UI](<@site/static/images/experiments/log_tables.png>)

There are many ways to use tables with rich media and interactive visualization. For more information, see [Tables Quickstart](../../data-vis/tables-quickstart) and the [Table data type](../../../ref/python/data-types/table).
