---
slug: /guides/weave
description: introducing Weave, the open-source, visual, interactive analytics toolkit for building AI
displayed_sidebar: default
---

# Weave

Weave is a visual development environment designed for building AI-powered software.  It is also an open-source, interactive analytics toolkit for performant data exploration.

Use Weave to:
* Spend less time waiting for datasets to load and more time exploring data, deriving insights, and building powerful data analytics
* Interactively explore your data. Work with your data visually and dynamically to discover patterns that static graphs can not reveal, without using complicated APIs.
* [Monitor AI applications and models in production](./prod-mon.md) with real-time metrics, customizable visualizations, and interactive analysis.
* Generate Boards to address common use cases when monitoring production models and working with LLMs
<!-- * W&B offers supplementary data & asset storage as well as a hosted compute engine -->


![](/images/weave/core_weave_demo.gif)

For more information about Weave, see the [Weave Github Repo](https://github.com/wandb/weave). To learn how to write your own queries interactively, see [this report](https://wandb.ai/luis_team_test/weave_example_queries/reports/Weave-queries---Vmlldzo1NzIxOTY2?accessToken=bvzq5hwooare9zy790yfl3oitutbvno2i6c2s81gk91750m53m2hdclj0jvryhcr).


## How it works
Use Weave to view your dataframe in your notebook with only a few lines of code:

1. First, install or update to the latest version of Weave with pip: 
```bash
pip install weave --upgrade
```
2. Load your dataframe into your notebook.
3. View your dataframe with `weave.show`. 

```python title="weave.ipynb" showLineNumbers
import weave
from sklearn.datasets import load_iris

# We load in the iris dataset for demonstrative purposes
iris = load_iris(as_frame=True)
df = iris.data.assign(target=iris.target_names[iris.target])

weave.show(df)
```

An interactive weave dashboard will appear, similar to the animation shown below:


![](/images/weave/first_load.gif)


The following animations show how you can interactively [plot charts](#plot-a-chart) and [publish your dashboard to share with your colleagues](#share-a-dashboard):

### Plot a chart
1. Hover your mouse next to a panel and click **Add a new panel**.
2. Copy the Weave Expression for the dataset you want to plot. This Weave Expression is the path/location of the dataset object in the Weave compute graph.
3. Click on Table to change this Weave Panel type.
4. From the dropdown, select **Plot**.

![](/images/weave/qs_table_plot.gif)

### Share a dashboard
Select the **Publish** button in the top right of your view to share your Weave Board:

![](/images/weave/make_quick_board.gif)

## How to get started

If this is your first time using Weave, we suggest that you explore the following topics:

* [Try Weave in a Jupyter notebook](https://colab.research.google.com/github/wandb/weave/blob/master/examples/get_started.ipynb).
* Explore the following topics:
   * [Stream data of any shape and type to W&B for analysis with Weave.](./streamtable.md)
   * [Create, share, and edit Weave Boards for analysis](./boards.md)
   * [Monitor AI applications and models in production](./prod-mon.md) with real-time metrics, customizable visualizations, and interactive analysis.
* Go to the Weave Home page at [weave.wandb.ai](https://weave.wandb.ai/) to see all of your Tables and Boards stored in W&B.
* See the [Weave Github Repo](https://github.com/wandb/weave).
