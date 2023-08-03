---
slug: /guides/tracer
description: Tools for the development of LLM-powered applications
displayed_sidebar: default
---

# Tracer for LLMs

:::info
Tracer is currently in preview and under active development.
:::

W&B Tracer is a suite of LLMOps tools built for the development of LLM-powered applications.
Use W&B Tracer to visualize and inspect the execution flow of your LLMs, analyze the inputs and outputs of your LLMs, view the intermediate results and securely store and manage your tracer and LLM chain configurations.

![](/images/tracer/trace_timeline.png)

W&B Tracer complements [W&B Experiments](../track/intro.md) and [W&B Tables](../tables/intro.md) to give an LLM developer everything they need to explore and experiment with confidence.

<!-- ## Tracer Product Suite

[Trace](#Trace) is the first of our Tracer tools -->

## How it works

W&B currently supports a tool called _Trace_. **Trace** allows you to track and visualize the inputs and outputs, execution flow, model architecture, and any intermediate results of your LLM chains. 

Use Trace for LLM chaining, plug-in or pipelining use cases. You can use your own LLM chaining implementation or use a W&B integration provided by LLM libraries such as LangChain.

Trace consists of three main components:

* [Trace table](#trace-table): Overview of the inputs and outputs of a chain.
* [Trace timeline](#trace-timeline): Displays the execution flow of the chain and is color-coded according to component types.
* [Model architecture](#model-architecture): View details about the structure of the chain and the parameters used to initialize each component of the chain.

### Trace Table
The Trace Table provides an overview of the inputs and outputs of a chain. The trace table also provides information about the composition of a trace event in the chain, whether or not the chain ran successfully, and any error messages returned when running the chain.

![](/images/tracer/trace_table.png)

Click on a row number on the left hand side of the Table to view the [Trace Timeline](#trace-timeline) for that instance of the chain.  

### Trace Timeline

The Trace Timeline view displays the execution flow of the chain and is color-coded according to component types. Select a trace event to display the inputs, outputs, and metadata of that trace.

![](/images/tracer/trace_timeline.png)

Trace events that raise an error are outlined in red. Click on a trace event colored in red to view the returned error message.

![](/images/tracer/trace_timeline_error.png)

### Model Architecture

The Model Architecture view provides details about the structure of the chain and the parameters used to initialize each component of the chain. Click on a trace event to learn more details about that event.

![](/images/tracer/model_architecture.png)

## How to get started

* If this is your first time using W&B Tracer, we recommend you go through the [Quickstart](./quickstart.md) guide.
* Try our [Google Colab Jupyter notebook](http://wandb.me/prompts-quickstart) for an example of how.

## More LLMs tools

Weights and Biases also has lightweight integrations for:
* [LangChain](../integrations/langchain.md)
* [OpenAI API](../integrations/other/openai.md)
* [Hugging Face Transformers](../integrations/huggingface.md)

<!-- Add link to colab -->

