---
slug: /guides/weave
description: introducing Weave, the open-source, visual, interactive analytics toolkit for building AI
displayed_sidebar: default
---

# Weave

## What is Weave?

Weave is a new kind of visual development environment, designed for building AI-powered software. It’s also an open-source, interactive analytics toolkit for performant data exploration.
Our mission is to equip machine learning practitioners with the best tools to turn data into insights quickly and easily. Whether you are a seasoned data scientist, an aspiring ML practitioner, or just a tech enthusiast eager to play with data, Weave is for you!

![](/images/weave/core_weave_demo.gif)

Learn more in the [Weave Github Repo→](https://github.com/wandb/weave)

* W&B offers supplementary data & asset storage as well as a hosted compute engine
* Stream data of any shape and type to W&B for analysis with Weave
* Create, share, and edit Weave Boards for analysis
* Generate Boards to address common use cases when monitoring production models and working with LLMs

## Primary workflows

Another way to think of Weave is as an ecosystem of computational patterns. We will highlight some of the core workflows here as they evolve:

* [production monitoring](./prod-mon.md): real-time metrics, custom visualizations, and interactive analysis for AI applications and deployed machine learning models serving users

## Weave quickstart

[Run in a Jupyter notebook→](https://github.com/wandb/weave/examples/experimental/skip_test/weave_demo_quickstart.ipynb)

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

## Why Weave

* **Performant:** Weave is built with performance in mind. It's designed to handle large datasets smoothly so you can focus on what matters - exploring data, deriving insights, and building powerful abstractions. Under the hood we optimize execution plans and parallelize computation using Arrow.
* **Interactive:** Weave enables an interactive flow state for data exploration. Work with your data visually and dynamically to discover patterns that static graphs can't reveal - without learning complicated APIs.
* **Modular ecosystem:** Weave's architecture & compute language consists of Types, Ops, and Panels. Combine, extend, and interconnect different components to build your custom data exploration toolkit. Share your creations to grow our collaborative ecosystem and help other contributors.
* **Open source:** We believe in the power of open source. Weave is built by the community, for the community. We are excited to see what you build with Weave!