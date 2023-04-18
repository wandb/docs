---
slug: /guides/prompts
description: Tools for the development of LLM-powered applications
---
# Prompts

Weights & Biases Prompts is a suite of LLMOps tools built for the development of LLM-powered applications.
Use W&B Prompts to visualize and inspect the execution flow of your LLMs, analyze the inputs and outputs of your LLMs, view the intermediate results and securely store and manage your prompts and LLM chain configurations.

![](/images/prompts/trace_timeline.png)

W&B Prompts complements [W&B Experiments](../track/intro.md) and [W&B Tables](../data-vis/intro.md) to give an LLM developer everything they need to explore and experiment with confidence.

<!-- ## Prompts Product Suite

[W&B Tracer](#Trace) is the first of our Prompts tools -->

## How it works

W&B currently supports a tool called _Tracer_. **Tracer** allows you to track and visualize the inputs and outputs, execution flow, model architecture, and any intermediate results of your LLM chains. 

Use Tracer for LLM chaining, plug-in or pipelining use cases. You can use your own LLM chaining implementation or use a W&B integration provided by LLM libraries such as LangChain.

W&B Tracer consists of three main components:

* [Trace table](#trace-table): Overview of the inputs and outputs of a chain
* [Trace timeline](#trace-timeline): Displays the execution flow of the chain and is color-coded according to component types.
* [Model architecture](#model-architecture): View details about the structure of the chain and the parameters used to initialize each component of the chain.

### Trace Table
The Trace Table provides an overview of the inputs and outputs of a chain. The trace table also provides information such the composition of components in the chain, whether or not the chain ran successfully, and any error messages when running the chain.

![](/images/prompts/trace_table.png)

To see more details for each row, click on the row number on the left hand side of the Table to see the Trace Timeline for that instance of the chain.

### Trace Timeline

The Trace Timeline view displays the execution flow of the chain and is color-coded according to component types. 

![](/images/prompts/trace_timeline.png)

Click on a component to display the inputs and outputs of that component along with metadata that was logged for that component.

![](/images/prompts/trace_timeline_detailed.png)


Components that raise an error are outlined in red. Click on a red component to view the returned error message.

![](/images/prompts/trace_timeline_error.png)

### Model Architecture

The Model Architecture view provides details about the structure of the chain and the parameters used to initialize each component of the chain. Click on a component to learn more details about that component.

![](/images/prompts/model_architecture.png)

## How to get started

* If this is your first time using W&B Prompts, we recommend you go through the [Quickstart](./quickstart.md) guide.
* Try our [Google Colab Jupyter notebook](http://wandb.me/prompts-quickstart) for an example of how [INSERT].

<!-- Add link to colab -->

