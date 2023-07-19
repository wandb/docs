---
slug: /guides/weave
description: introducing Weave, the open-source, visual, interactive analytics toolkit for building AI
displayed_sidebar: default
---

# Weave

## What is Weave?

Weave is a visual development environment designed for building AI-powered software.  It is also an open-source, interactive analytics toolkit for performant data exploration and a growing ecosystem of computational patterns. 

Use Weave to:
* Spend less time waiting for datasets to load and more time exploring data, deriving insights, and building powerful data analytics
* Interactively explore your data. Work with your data visually and dynamically to discover patterns that static graphs can not reveal, without using complicated APIs.
* [Monitor AI applications and models in production](./prod-mon.md) with real-time metrics, customizable visualizations, and interactive analysis.

![](/images/weave/core_weave_demo.gif)

Learn more in the [Weave Github Repo→](https://github.com/wandb/weave)

* W&B offers supplementary data & asset storage as well as a hosted compute engine
* Stream data of any shape and type to W&B for analysis with Weave
* Create, share, and edit Weave Boards for analysis
* Generate Boards to address common use cases when monitoring production models and working with LLMs

## How it works

[Try Weave in a Jupyter notebook→](https://github.com/wandb/weave/blob/master/examples/experimental/skip_test/weave_demo_quickstart.ipynb)

* install via `pip install weave`
* `import weave` in a notebook
* explore your data  with one line of code!

### View a dataframe

```python
import weave
from sklearn.datasets import load_iris

# use any existing dataframe
# here we load the iris data and visualize the labels
iris = load_iris(as_frame=True)
df = iris.data.assign(target=iris.target_names[iris.target])

weave.show(df)
```

![](/images/weave/first_load.gif)

### Plot a chart

![](/images/weave/qs_table_plot.gif)

### Share a dashboard

![](/images/weave/make_quick_board.gif)