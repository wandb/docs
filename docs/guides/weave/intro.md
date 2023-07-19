---
slug: /guides/weave
description: the extension toolkit for W&B
displayed_sidebar: default
---

# What is Weave?

Weave is a new kind of visual development environment, designed for building AI-powered software. It’s also an open-source, interactive analytics toolkit for performant data exploration.
Our mission is to equip machine learning practitioners with the best tools to turn data into insights quickly and easily. Whether you are a seasoned data scientist, an aspiring ML practitioner, or just a tech enthusiast eager to play with data, Weave is for you!

Learn more in the [Weave Github Repo→](https://github.com/wandb/weave)

* W&B offers supplementary data & asset storage as well as a hosted compute engine
* Stream data of any shape and type to W&B for analysis with Weave
* Create, share, and edit boards for analysis
* Generate boards to address common use cases when monitoring production models and working with LLMs.

## Primary workflows

Another way to think of Weave is as an ecosystem of computational patterns. We will highlight some of the core workflows here as they evolve:

* production monitoring

# Weave quickstart

[Run in a Jupyter notebook→](https://github.com/wandb/weave/examples/experimental/skip_test/weave_demo_quickstart.ipynb)

* install via `pip install weave`
* `import weave` in a notebook
* explore your data  with one line of code!

## 1. View a dataframe

## 2. Add a plot

## 3. Create and share dashboards

# Why Weave

* **Performant:** Weave is built with performance in mind. It's designed to handle large datasets smoothly so you can focus on what matters - exploring data and finding insights. Under the hood we optimize execution plans and parallelize computation using Arrow.
* **Interactive:** Weave is all about making data exploration fun and interactive. It empowers you to engage with your data and discover patterns that static graphs can't reveal - without learning complicated APIs! Beautiful and interactive plots to bring your data to life.
* **Modular Ecosystem:** Weave's architecture & compute language is build on Types, Ops, and Panels. Combine different components to build your customized data exploration toolkit, and publish reusable components into the ecosystem for others to use!
* **Open-Source:** We believe in the power of open-source. Weave is built by the community, for the community. We are excited to see how you use it and what you build with it.