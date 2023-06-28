---
description: The Prompts Quickstart shows how to visualise and debug the execution flow of your LLM chains and pipelines
displayed_sidebar: default
---


# Prompts Quickstart

[**Try in a Colab Notebook here →**](http://wandb.me/prompts-quickstart)

<head>
  <title>Prompts Quickstart</title>
</head>

This Quickstart guide will walk you how to use [Trace](intro.md) to visualize and debug calls to LangChain, LlamaIndex or your own LLM Chain or Pipeline:

1. **[Langchain:](#use-trace-with-langchain)** Use the 1-line LangChain environment variable or context manager integration for automated logging.

2. **[LlamaIndex:](#use-trace-with-llamaindex)** Use the W&B callback from LlamaIndex for automated logging.

3. **[Custom usage](#use-trace-with-any-llm-chain-or-plug-in)**: Use Trace with your own custom chains and LLM pipeline code.


## Use W&B Trace with LangChain

:::info
**Versions** Please use `wandb >= 0.15.4` and `langchain >= 0.0.218`
:::

With a 1-line environment variable from LangChain, W&B Trace will continuously log calls to a LangChain Model, Chain, or Agent. 

Note that you can also see the documentation for W&B Trace in the [LangChain documentation](https://python.langchain.com/docs/ecosystem/integrations/agent_with_wandb_tracing).

For this quickstart, we will use a LangChain Math Agent:

### 1. Set the LANGCHAIN_WANDB_TRACING environment variable

First, set the LANGCHAIN_WANDB_TRACING environment variable to true. This will turn on automated Weights & Biases logging with LangChain:

```python
import os

# turn on wandb logging for langchain
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
```

Thats it! Now any call to a LangChain LLM, Chain, Tool or Agent will be logged to Weights & Biases.

### 2. Configure your Weights & Biases settings
You can optionally set additional Weights & Biases [Environment Variables](/guides/track/environment-variables) to set parameters that are typically passed to `wandb.init()`. Parameters often used include `WANDB_PROJECT` or `WANDB_ENTITY` for more control over where your logs are sent in W&B. For more information about [`wandb.init`](../../ref/python/init.md), see the API Reference Guide.

```python
# optionally set your wandb settings or configs
os.environ["WANDB_PROJECT"] = "langchain-tracing"
```


### 3. Create a LangChain Agent
Create a standard math Agent using LangChain:

```python
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType

llm = ChatOpenAI(temperature=0)
tools = load_tools(["llm-math"], llm=llm)
math_agent = initialize_agent(tools, 
                              llm, 
                              agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
```


### 4. Run the Agent and start Weights & Biases logging
Use LangChain as normal by calling your Agent. You will see a Weights & Biases run start and be asked for your Weights & Biases **[API key](https:wwww.wandb.ai/authorize)**. Once your enter your API key, the inputs and outputs of your Agent calls will start to be streamed to the Weights & Biases App.

```python
# some sample maths questions
questions = [
  "Find the square root of 5.4.",
  "What is 3 divided by 7.34 raised to the power of pi?",
  "What is the sin of 0.47 radians, divided by the cube root of 27?"
]

for question in questions:
  try:
    # call your Agent as normal
    answer = math_agent.run(question)
    print(answer)
  except Exception as e:
    # any errors will be also logged to Weights & Biases
    print(e)
    pass
```

Once each Agent execution completes, all calls in your LangChain object will be logged to Weights & Biases


### 5. View the trace in Weights & Biases

Click on the W&B [run](../runs/intro.md) link generated in the previous step. This will redirect you to your Project workspace in the W&B App. 

Select a run you created to view the trace table, trace timeline and the model architecture of your LLM. 

![](/images/prompts/trace_timeline_detailed.png)


### 6. LangChain Context Manager
Depending on your use case, you might instead prefer to use a context manager to manage your logging to W&B:

```python
from langchain.callbacks import wandb_tracing_enabled

# unset the environment variable and use a context manager instead
if "LANGCHAIN_WANDB_TRACING" in os.environ:
    del os.environ["LANGCHAIN_WANDB_TRACING"]

# enable tracing using a context manager
with wandb_tracing_enabled():
    math_agent.run("What is 5 raised to .123243 power?")  # this should be traced

math_agent.run("What is 2 raised to .123243 power?")  # this should not be traced
```


Please report any issues with this LangChain integration to the [wandb repo](https://github.com/wandb/wandb/issues) with the tag `langchain`

## Use W&B Trace with LlamaIndex

:::info
**Versions** Please use `wandb >= 0.15.4` and `llama-index >= 0.6.35`
:::

At the lowest level, LlamaIndex uses the concept of start/end events ([`CBEventTypes`](https://gpt-index.readthedocs.io/en/latest/reference/callbacks.html#llama_index.callbacks.CBEventType)) to keep a track of logs. Each event has some payload which provides information like, the query asked and the response generated by the LLM, or about the number of documents used to create N chunks, etc.

At a higher level, they have recently introduced the concept of Callback Tracing which builds a trace map of connected events. For example when you query over an index, under the hood, retrieval, LLM calls, etc. takes place.

The `WandbCallbackHandler` provides an intuitive way to visualize and track this trace map. It captures the payload of the events and logs them to wandb. It also tracks necessary metadata like total token counts, prompt, context, etc.

Moreover, this callback can also be used to upload and download indices to/from W&B Artifacts for version controlling your indices.

### 1. Import WandbCallbackHandler

First import the `WandbCallbackHandler` and set it up. You can also pass additional parameters [`wandb.init`](../../ref/python/init.md) parameteres such as your W&B Project or Entity.

You will see a W&B run start and be asked for your Weights & Biases **[API key](https:wwww.wandb.ai/authorize)**. A W&B run link will be generated, here you'll be able to view your logged LlamaIndex queries and data once you start logging.

```python
from llama_index import ServiceContext
from llama_index.callbacks import CallbackManager, WandbCallbackHandler

# initialise WandbCallbackHandler and pass any wandb.init args
wandb_args = {"project":"llamaindex"}
wandb_callback = WandbCallbackHandler(run_args=wandb_args)

# pass wandb_callback to the service context
callback_manager = CallbackManager([wandb_callback])
service_context = ServiceContext.from_defaults(callback_manager=callback_manager)
```

### 2. Build an Index

We will build a simple index using a text file.

```python
docs = SimpleDirectoryReader("path_to_dir").load_data()
index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context)
```

### 3. Query an index and start Weights & Biases logging

With the loaded index, start querying over your documents. Every call to your index will be automatically logged to Weights & Biases

```python
questions = [
  "What did the author do growing up?",
  "Did the author travel anywhere?",
  "What does the author love to do?"
]

query_engine = index.as_query_engine()

for q in questions:
  response = query_engine.query(q)
```

### 4. View the trace in Weights & Biases

Click on the Weights and Biases run link generated while initializing the `WandbCallbackHandler` in step 1. This will take you to your project workspace in the W&B App where you will find a trace table and a trace timeline.

![](/images/prompts/llama_index_trace.png)

### 5. Finish tracking

When you are done tracking your LLM queries, it is good practice to close the wandb process like so:

```python
wandb_callback.finish()
```

Thats it! Now you can log your queries to your index using Weights & Biases. If you come across any issues, please file an issue on the [wandb repo](https://github.com/wandb/wandb/issues) with the tag `llamaindex`

### 6. [Optional] Save your Index data in Weights & Biaes Artifacts
Weights & Biases [Artifacts](guides/artifacts) is a versioned data and model storage product. 

By logging your index to Artifacts and then using it when needed, you can assosciate a particular version of your index with the logged Trace outputs, ensuring full visibility into what data was in your index for a particular call to your index.


```python
#  The string passed to the `index_name` will be your artifact name
wandb_callback.persist_index(index, index_name="my_vector_store")
```

You can then go to the artifacts tab on your W&B run page to view the uploaded index.

**Using an Index stored in W&B Artifacts**

When you load an index from Artifacts you'll return a [`StorageContext`](https://gpt-index.readthedocs.io/en/latest/reference/storage.html). Use this storage context to load the index into memory using a function from the LlamaIndex [loading functions](https://gpt-index.readthedocs.io/en/latest/reference/storage/indices_save_load.html).


```python
from llama_index import load_index_from_storage

storage_context = wandb_callback.load_storage_context(artifact_url="<entity/project/index_name:version>")
index = load_index_from_storage(storage_context, service_context=service_context)
```

**Note:** For a [`ComposableGraph`](https://gpt-index.readthedocs.io/en/latest/reference/query/query_engines/graph_query_engine.html) the root id for the index can be found in the artifact's metadata tab in the W&B App.


## Use Trace with Any LLM Chain or Plug-In

When logging with Trace, a single run can have multiple calls to a LLM, Tool, Chain or Agent logged to it, there is no need to start a new run after each generation from your model, each call will be appended to the Trace Table.

To use Trace with your own chains, plug-ins or pipelines, you first need to create traces using the `Span` and `TraceTree` data types. A _Span_ represents a unit of work.

### 1. Create a Span
First, create a span object. Import `trace_tree` from the `wandb.sdk.data_types`:

```python
from wandb.sdk.data_types import trace_tree

# Root Span - Create a span for your high level agent
root_span = trace_tree.Span(name="Auto-GPT", 
  span_kind = trace_tree.SpanKind.AGENT)
```

Spans can be of type `AGENT`, `CHAIN`, `TOOL` or `LLM`

### 2. Add child Spans
Nest child Spans within the parent span so that they are nested and in the correct order in the Trace Timeline view. 

The following text code demonstrates how to create two child spans and one grandchild span:

```python
tool_span = trace_tree.Span(
  name="Tool 1", span_kind = trace_tree.SpanKind.TOOL
)

chain_span = trace_tree.Span(
  name="LLM CHAIN 1", span_kind = trace_tree.SpanKind.CHAIN
)

llm_span = trace_tree.Span(
  name="LLM 1", span_kind = trace_tree.SpanKind.LLM
)

chain_span.add_child_span(llm_span)
root_span.add_child_span(tool_span)
root_span.add_child_span(chain_span)
```

### 3. Add the inputs and outputs

Populate spans with the input and output data as well as any metadata: 

```python
# add the Inputs and Outputs to the span as dictionaries
tool_span.add_named_result(
  {"input": "search: google founded in year"}, 
  {"response": "1998"}
)

chain_span.add_named_result(
  {"input": "calculate: 2023 - 1998"}, 
  {"response": "25"}
)

llm_span.add_named_result(
  {"system": "you are a helpful assistant", 
    "input": "calculate: 2023 - 1998"}, 
  {"response": "25"}
)

root_span.add_named_result(
  {"user": "How old is google?"},
  {"response": "25 years old"}
)
```

### 4. Add metadata, status, start and end time to a Span

Any span can also have metadata, status, status messages, start and end timestamps:

```python
# add metadata to the span using .attributes
tokens_used = 284
llm_span.attributes = {"token_usage": tokens_used}

# often you want to add the same metadata to different spans
root_span.attributes = {"token_usage": tokens_used}

# add a status code and any message to any span
root_span.status_code = trace_tree.StatusCode.ERROR  # or SUCCESS
root_span.status_message = "Error: there was an error"

# add the start and end timestamp for any span, in milliseconds 
root_span.start_time_ms = 1685649600011
root_span.end_time_ms = 1685649611000
```

### 5. Log the spans to W&B Trace 

Log your span to W&B with the run.log() method. W&B will create a Trace Table and Trace Timeline for you to view in the W&B App UI.


```python
import wandb 

trace = trace_tree.WBTraceTree(root_span)
run = wandb.init(project="wandb_prompts")
run.log({"trace": trace})
run.finish()
```
### 6. View the trace
Click on the W&B run link that is generated to see the trace of your LLM on the W&B App UI.