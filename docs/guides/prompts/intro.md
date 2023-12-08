---
slug: /guides/prompts
description: Tools for the development of LLM-powered applications
displayed_sidebar: default
---
# Prompts for LLMs

W&B Prompts is a suite of LLMOps tools built for the development of LLM-powered applications. Use W&B Prompts to visualize and inspect the execution flow of your LLMs, analyze the inputs and outputs of your LLMs, view the intermediate results and securely store and manage your prompts and LLM chain configurations.

## Use Cases

W&B Prompts provides several solutions for building and monitoring LLM-based apps. Software developers, prompt engineers, ML practitioners, data scientists, and other stakeholders working with LLMs need cutting-edge tools to:

- [Explore and debug LLM chains](https://docs.wandb.ai/guides/prompts) and prompts with greater granularity.
- Monitor and observe LLMs to better understand and evaluate performance, usage, and budgets.

## Products

### Traces

W&B’s LLM tool is called *Traces*. **Traces** allow you to track and visualize the inputs and outputs, execution flow, model architecture, and any intermediate results of your LLM chains.

Use Traces for LLM chaining, plug-in or pipelining use cases. You can use your own LLM chaining implementation or use a W&B integration provided by LLM libraries such as LangChain.

Traces consists of three main components:

- [Trace table](https://docs.wandb.ai/guides/prompts#trace-table): Overview of the inputs and outputs of a chain.
- [Trace timeline](https://docs.wandb.ai/guides/prompts#trace-timeline): Displays the execution flow of the chain and is color-coded according to component types.
- [Model architecture](https://docs.wandb.ai/guides/prompts#model-architecture): View details about the structure of the chain and the parameters used to initialize each component of the chain.

**Trace Table**

The Trace Table provides an overview of the inputs and outputs of a chain. The trace table also provides information about the composition of a trace event in the chain, whether or not the chain ran successfully, and any error messages returned when running the chain.

![Screenshot of a trace table.](/images/prompts/trace_table.png)

Click on a row number on the left hand side of the Table to view the Trace Timeline for that instance of the chain.

**Trace Timeline**

The Trace Timeline view displays the execution flow of the chain and is color-coded according to component types. Select a trace event to display the inputs, outputs, and metadata of that trace.

![Screenshot of a Trace Timeline.](/images/prompts/trace_timeline.png)

Trace events that raise an error are outlined in red. Click on a trace event colored in red to view the returned error message.

![Screenshot of a Trace Timeline error.](/images/prompts/trace_timeline_error.png)

**Model Architecture**

The Model Architecture view provides details about the structure of the chain and the parameters used to initialize each component of the chain. Click on a trace event to learn more details about that event.

### Weave

Weave is a visual development environment designed for building AI-powered software. It is also an open-source, interactive analytics toolkit for performant data exploration.

Use Weave to:

- Spend less time waiting for datasets to load and more time exploring data, deriving insights, and building powerful data analytics
- Interactively explore your data. Work with your data visually and dynamically to discover patterns that static graphs can not reveal, without using complicated APIs.
- [Monitor AI applications and models in production](https://docs.wandb.ai/guides/weave/prod-mon) with real-time metrics, customizable visualizations, and interactive analysis.
- Generate Boards to address common use cases when monitoring production models and working with LLMs.

### How it works

Use Weave to view your dataframe in your notebook with only a few lines of code:

1. First, install or update to the latest version of Weave with pip:

```bash
pip install weave --upgrade
```

1. Load your dataframe into your notebook.
2. View your dataframe with `weave.show(df)`.

```python
import weave
from sklearn.datasets import load_iris

# We load in the iris dataset for demonstrative purposes
iris = load_iris(as_frame=True)
df = iris.data.assign(target=iris.target_names[iris.target])

weave.show(df)
```

An interactive weave dashboard will appear, similar to the animation shown below:

![https://docs.wandb.ai/assets/images/first_load-af370d0fcdf6ced0334c2bfde5871165.gif](https://docs.wandb.ai/assets/images/first_load-af370d0fcdf6ced0334c2bfde5871165.gif)

## Integrations

Weights and Biases also has lightweight integrations for:

- [LangChain](https://docs.wandb.ai/guides/integrations/langchain)
- [OpenAI API](https://docs.wandb.ai/guides/integrations/openai-api)
- [OpenAI GPT-3.5 Fine-Tuning](https://docs.wandb.ai/guides/integrations/openai)
- [Hugging Face Transformers](https://docs.wandb.ai/guides/integrations/huggingface)

## Getting Started

We recommend you go through the Prompts [Quickstart](https://docs.wandb.ai/guides/prompts/quickstart) guide, which will walk you through logging a custom LLM pipeline with Trace. A [colab](http://wandb.me/prompts-quickstart) version of the guide is also available. 

## Next Steps

- Check out more detailed documentation on [Weave](https://github.com/wandb/weave/tree/master/examples), [Trace](https://colab.research.google.com/github/wandb/weave/blob/master/examples/prompts/trace_debugging/trace_quickstart_langchain.ipynb), or our [OpenAI](https://docs.wandb.ai/guides/prompts/openai) Integration.
- Try one of our [demo colabs](https://github.com/wandb/weave/tree/master/examples), which offer more detailed explanations of how to use Prompts for LLM ops and building interactive data applications.