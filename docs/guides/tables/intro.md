---
slug: /guides/tables
description: Iterate on datasets and understand model predictions
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Visualize your data

<CTAButtons productLink="https://wandb.ai/wandb/examples/reports/AlphaFold-ed-Proteins-in-W-B-Tables--Vmlldzo4ODc0MDc" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Tables_Quickstart.ipynb"/>

Use W&B Tables to visualize and query tabular data.

* Look at sample model predictions visually
* Query to find commonly misclassified examples
* Compare how different models perform on the same test set
* Identify higher level patterns in your data, like an under-represented class


Here is [an example Table](https://wandb.ai/av-team/mlops-course-001) with interactive semantic segmentation and calculated metrics.
![](/images/data_vis/tables_sample_predictions.png)

## How it works

A W&B Table is a two dimensional grid of data where each column has a single type of data. Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types. 

Log a table with a few lines of code:

1. Create a W&B Run with [`wandb.init()`](../../ref/python/init.md). 
2. Create a [`wandb.Table()`](../../ref/python/data-types/table.md) object instance. Pass the name of the columns in your table along with the data for the `columns` and `data` parameters, respectively.  
3. Log the table with [`run.log()`](../../ref/python/log.md) as a key-value pair. Provide a name for your table for the key, and pass the object instance of `wandb.Table` as the value.

```python showLineNumbers
run = wandb.init(project="table-test")
my_table = wandb.Table(
    columns=["a", "b"], 
    data=[["a1", "b1"], ["a2", "b2"]]
    )
run.log({"Table Name": my_table})   
```



## How to get started
<!-- * If this is your first time using W&B Artifacts, we recommend you go through the [Artifacts Colab notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Artifacts_Quickstart_with_W&B.ipynb#scrollTo=fti9TCdjOfHT). -->
* Read the [quickstart](./tables-quickstart.md) for a step-by-step instructions on how to log data tables, visualize data, and query data.
* View the [Tables gallery](./tables-gallery.md) for different ways you can use tables.

