---
slug: /guides/weave/prod-mon
description: production monitoring with Weave
displayed_sidebar: default
---

# Production monitoring

Production monitoring for AI means real-time observability and analytics for any models served from your application. For models deployed to production, monitoring tools and dashboards help track key performance metrics like query rates and latency and enable interactive analytics around model prediction quality and trends, patterns of errors or edge cases, data drift, etc.

# How do I get started using W&B to monitor models?

W&B offers a data management service to compliment the open source [Weave](http://github.com/wandb/weave) project. You can stream live data (and/or save batch tables) in any schema that makes sense for your use case and workflow—like a no-setup database, without the SQL. This approach is effective for tracking and visualizing live production queries, model predictions, dynamic evaluation metrics, user feedback, and more. 

Get started in two steps:

1. Log data using the Weave StreamTable API
2. Seed a Weave Board from the UI

# Weave StreamTable API & Usage
```python
from weave.monitoring import StreamTable
table = StreamTable("prodmon_demo")
for i in range(100):
  table.log({"_id" : i, "text" : "hi " + str(i) + "!" , "img" : gen_image()}
```

### StreamTables features

* Persisted and secured in W&B
* Columnar storage for efficient queries
* Any arbitrary data shape
* Custom, non-primitive types (e.g, images)
* Supports multiple parallel clients writers
* Automatically track log time

Read about the [Weave StreamTable API →](https://github.com/wandb/weave/blob/master/examples/experimental/ProductionMonitoring/StreamTable.md)

Try a simple interactive example [in a Jupyter notebook→](https://github.com/wandb/weave/blob/master/examples/experimental/ProductionMonitoring/stream_table_api.ipynb)

## Weave Monitor Decorator
```python
from weave.monitoring import monitor

@monitor()
def ask_llm_calculator(prompt, question):
	return agent.run(prompt + " " + question)

ask_llm_calculator(
	"Please accurately answer the following question:",
	"Find the square root of 5.4"
)
```

### Weave Monitor Decorator features

* tracks inputs, outputs, latency, timestamp, & exceptions.
* supports pre- and post- processing of inputs and outputs
* able to add data to rows after execution

Try an interactive example in a Jupyter notebook——monitor an MNIST model [with live user feedback→](https://github.com/wandb/weave/blob/master/examples/experimental/ProductionMonitoring/ProductionMonitoringConceptualOverview.ipynb)

# Seeding a Weave Board

You can save and customize any StreamTable in the UI as a Weave Board:

1. Starting from a StreamTable Weave Panel in the UI, open the panel in a new tab to show a full screen dashboard. 
2. Click “seed new board” in the bottom right to create a new Weave Board.
3. Optionally customize the Board: edit the current SteamTable panel, add other panels, rename the Board, etc.
4. Click “commit” to save your local changes and “publish” to share your Board.