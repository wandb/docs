---
displayed_sidebar: tutorials
title: Iterate on LLMs
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

<CTAButtons colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/prompts/WandB_Prompts_Quickstart.ipynb"></CTAButtons>

**Weights & Biases Prompts** is a suite of LLMOps tools built for the development of LLM-powered applications. 

Use W&B Prompts to visualize and inspect the execution flow of your LLMs, analyze the inputs and outputs of your LLMs, view the intermediate results and securely store and manage your prompts and LLM chain configurations.

## Installation


```python
!pip install "wandb==0.15.2" -qqq
```
# W&B Prompts

W&B currently supports a tool called __Trace__. Trace consists of three main components:

**Trace table**: Overview of the inputs and outputs of a chain.

**Trace timeline**: Displays the execution flow of the chain and is color-coded according to component types.

**Model architecture**: View details about the structure of the chain and the parameters used to initialize each component of the chain.

In the proceeding image you see a new panel automatically created in your workspace, showing each execution, the trace, and the model architecture


{{< img src="/images/tutorials/prompts_quickstart/prompts.png" alt="prompts_1" >}}

{{< img src="/images/tutorials/prompts_quickstart/prompts2.png" alt="prompts_2" >}}

# Writing your own integration

What if you want to write an integration or instrument your code? This is where utilities like `TraceTree` and `Span` comes in handy.

{{< img src="/images/tutorials/prompts_quickstart/prompts3.png" alt="prompts_3" >}}

**Note:** W&B Runs support logging as many traces you needed to a single run, i.e. you can make multiple calls of `run.log` without the need to create a new run each time


```python
from wandb.sdk.data_types import trace_tree
import wandb
```

A Span represents a unit of work, Spans can have type `AGENT`, `TOOL`, `LLM` or `CHAIN`


```python
parent_span = trace_tree.Span(
  name="Example Span", 
  span_kind = trace_tree.SpanKind.AGEN
)
```

Spans can (and should!) be nested:


```python
# Create a span for a call to a Tool
tool_span = trace_tree.Span(
  name="Tool 1", 
  span_kind = trace_tree.SpanKind.TOOL
)

# Create a span for a call to a LLM Chain
chain_span = trace_tree.Span(
  name="LLM CHAIN 1", 
  span_kind = trace_tree.SpanKind.CHAIN
)

# Create a span for a call to a LLM that is called by the LLM Chain
llm_span = trace_tree.Span(
  name="LLM 1", 
  span_kind = trace_tree.SpanKind.LLM
)
chain_span.add_child_span(llm_span)
```

Span Inputs and Outputs can be added like so:


```python
tool_span.add_named_result(
  {"input": "search: google founded in year"}, 
  {"response": "1998"}
)
chain_span.add_named_result(
  {"input": "calculate: 2023 - 1998"}, 
  {"response": "25"}
)
llm_span.add_named_result(
  {"input": "calculate: 2023 - 1998", "system": "you are a helpful assistant", }, 
  {"response": "25", "tokens_used":218}
)

parent_span.add_child_span(tool_span)
parent_span.add_child_span(chain_span)

parent_span.add_named_result({"user": "calculate: 2023 - 1998"}, 
                             {"response": "25 years old"})
```

You can then log the parent_span to W&B like as below. 


```python
run = wandb.init(name="manual_span_demo", project="wandb_prompts_demo")
run.log({"trace": trace_tree.WBTraceTree(parent_span)})
run.finish()
```

Clicking on the W&B Run link generated will take you to a workspace where you can inspect the Trace created.
