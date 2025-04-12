---
title: plot_table
object_type: api
---

{{< cta-button githubLink=https://github.com/wandb/wandb/blob/main/wandb/wandb/plot/custom_chart.py >}}




### <kbd>function</kbd> `plot_table`

```python
plot_table(
    vega_spec_name: 'str',
    data_table: 'wandb.Table',
    fields: 'dict[str, Any]',
    string_fields: 'dict[str, Any] | None' = None,
    split_table: 'bool' = False
) → CustomChart
```

Creates a custom charts using a Vega-Lite specification and a `wandb.Table`. 

This function creates a custom chart based on a Vega-Lite specification and a data table represented by a `wandb.Table` object. The specification needs to be predefined and stored in the W&B backend. The function returns a custom chart object that can be logged to W&B using `wandb.log()`. 



**Args:**
 
 - `vega_spec_name`:  The name or identifier of the Vega-Lite spec  that defines the visualization structure. 
 - `data_table`:  A `wandb.Table` object containing the data to be  visualized. 
 - `fields`:  A mapping between the fields in the Vega-Lite spec and the  corresponding columns in the data table to be visualized. 
 - `string_fields`:  A dictionary for providing values for any string constants  required by the custom visualization. 
 - `split_table`:  Whether the table should be split into a separate section  in the W&B UI. If `True`, the table will be displayed in a section named  "Custom Chart Tables". Default is `False`. 



**Returns:**
 
 - `CustomChart`:  A custom chart object that can be logged to W&B. To log the  chart, pass it to `wandb.log()`. 



**Raises:**
 
 - `wandb.Error`:  If `data_table` is not a `wandb.Table` object. 
