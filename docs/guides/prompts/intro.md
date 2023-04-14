---
slug: /guides/prompts
description: Tools for the development of LLM-powered applications
---
# Prompts

Weights & Biases Prompts is a suite of LLMOps tools built for the development of LLM-powered applications. It encompasses tooling to visualise and inspect the execution flow of your LLMs, analyse the inputs and outputs of your LLMs, view the itermediate results and securely store and manage your prompts and LLM chain configurations. 

W&B Prompts complements [W&B Experiments](../track/intro.md) and [W&B Tables](../data-vis/intro.md) to give an LLM developer everything they need to explore and experiment with confidence.

![](/images/prompts/trace_timeline.png)

## Prompts Product Suite

[W&B Tracer](#Trace) is the first of our Prompts tools

### Tracer
**Tracer** is a tool that allows you to track and visualize the inputs and outputs, execution flow, model architecture, and any intermediate results of your LLM chains. Tracer can be used for any LLM chaining, plug-in or pipelining usecase - either using your own LLM chaining implementation or using one of the Weights & Biases integrations in other LLM libraries such as LangChain. The Tracer is composed of Trace Table, Trace Timeline and Model Architecture views.

#### Trace Table
The Trace Table provides an overview of the inputs and outputs of the chain as well as other information such as the composition of compentents in the chain, whether or not the chain ran succesfully, and any error messages when running the chain. To see more detail for each row, click on the row number on the left hand side of the Table to see the Trace Timeline for that instance of the chain.

![](/images/prompts/trace_table.png)

#### Trace Timeline

The Trace Timeline view displays the execution flow of the chain and is color-coded according to component types. 

![](/images/prompts/trace_timeline.png)

Clicking on a component displays the inputs and outputs to that component as well as additional metadata provided. 

![](/images/prompts/trace_timeline_detailed.png)

If an error is raised, the component that raised the error will be outlined in red and the full error message displayed

![](/images/prompts/trace_timeline_error.png)

#### Model Architecture

The Model Archtecture view provides details about the structure of the chain and the parameters used to initialise each component of the chain. Clicking on each component provides more details about it.

![](/images/prompts/model_architecture.png)

## How to get started

See the Prompts [Quickstart](./quickstart.md) guide to get started
