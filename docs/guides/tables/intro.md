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

A Table is a two-dimensional grid of data where each column has a single type of data. Tables support primitive and numeric types, as well as nested lists, dictionaries, and rich media types. 

Log a table with a few lines of code:

- [`wandb.init()`](../../ref/python/init.md): Create a [run](/guides/runs.md) to track results.
- [`wandb.Table()`](../../ref/python/data-types/table.md): Create a new table object.
  - `columns`: Set the column names.
  - `data`: Set the contents of the table.
- [`run.log()`](../../ref/python/log.md): Log the table to save it to W&B.

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
* [Quickstart](./tables-quickstart.md): Learn to log data tables, visualize data, and query data.
* [Tables Gallery](./tables-gallery.md): See example use cases for Tables.

