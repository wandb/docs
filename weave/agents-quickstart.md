---
title: "Quickstart: Trace an agent"
description: Trace a multi-turn agent with the Weave SDK. Sessions, turns, LLM calls, and tool calls render in the Agents view of your project.
---

[Try in Colab](https://colab.research.google.com/github/wandb/weave/blob/master/docs/weave/cookbooks/source/agents-quickstart.ipynb) · [GitHub source](https://github.com/wandb/weave/blob/master/docs/weave/cookbooks/source/agents-quickstart.ipynb)

The Weave SDK allows you to trace custom agents or agents created using popular SDKs. This quickstart guides you through how to manually integrate Weave into a custom-built multi-turn agent to emit and capture OpenTelemetry spans and render them in Weave's Agents view.

If you are looking to integrate Weave with popular SDKs or harnesses, such as the Claude Agents SDK or Codex, see the [Weave integration section](/weave/guides/integrations). Weave autopatches into several popular agent-building SDKs and agent harnesses for quick integration.

## What you'll learn

The code in this guide sets up a small research agent that can look things up on Wikipedia. It asks three questions (three turns), lets the AI decide when to search Wikipedia for an answer, and uses Weave to record every step (the conversation, each question, each AI response, and each Wikipedia lookup) so you can see exactly what happened in the Weave Agents view.

This guide shows you how to:

- Initialize Weave for agent tracing with `weave.init()`
- Open a session and a turn with `weave.start_session()` and `weave.start_turn()`
- Wrap LLM calls with `weave.start_llm()` and record usage
- Wrap tool executions with `weave.start_tool()` and record results
- View the resulting session, turns, and tool calls in the Agents view

## How the Weave SDK works with agents

The Weave SDK includes a generic OTel ingest system for agents, meaning that Weave can capture information from any OTel span in your agent's code. However, Weave requires special handling of the following spans to render your agent's traces in the Agents view of the Weave UI.

| Function | Maps to | OTel span |
| --- | --- | --- |
| `weave.start_session(...)` | A conversation | (no span — groups turns) |
| `weave.start_turn(...)` | One user / agent exchange | `invoke_agent` |
| `weave.start_llm(...)` | One LLM API call | `chat` |
| `weave.start_tool(...)` | One tool execution | `execute_tool` |

All four are context managers. On exit, they end the span and flush attributes, including on exceptions.

Other [GenAI semantic-convention attributes](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/), such as `gen_ai.usage.*` and `gen_ai.agent.name`, enable additional rendering, but they are optional.

## Prerequisites

- A W&B account and [API key](https://wandb.ai/authorize)
- Python 3.10+
- An OpenAI API key

## Install packages

Install the following packages into your developer environment:

```bash
pip install weave openai requests
```

## Initialize Weave

`weave.init()` authenticates with W&B and configures the OTel exporter that sends agent spans to the **Agents** view. If the project does not exist on your team, Weave creates it the first time you write to it.

```python
import os

os.environ["WANDB_API_KEY"] = "your-wandb-api-key"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

TEAM = "<your-team-name>"
PROJECT = "<your-project-name>"

import weave
weave.init(f"{TEAM}/{PROJECT}")
```

## Define a tool

The following code defines the agent's Wikipedia search tool and an OpenAI tool schema to determine when and how to use the tool.

```python
import json
import requests

def wikipedia_search(query: str) -> str:
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
            "prop": "extracts", "exintro": True, "explaintext": True, "format": "json",
        },
        headers={"User-Agent": "weave-demo"},
    ).json()
    return next(iter(r["query"]["pages"].values()))["extract"]

wikipedia_tool_schema = {
    "type": "function",
    "function": {
        "name": "wikipedia_search",
        "description": "Search Wikipedia for a topic and return its intro paragraph.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
```

## Run a traced multi-turn agent

The example below runs three turns in a single session. Each turn:

1. Opens a `chat` span and lets the LLM decide whether to call the tool
2. If the LLM requested a tool, opens an `execute_tool` span around the call and feeds the result back to the LLM
3. Opens a second `chat` span to produce the final answer

```python
from openai import OpenAI

openai_client = OpenAI()
MODEL = "gpt-4o-mini"

def run_turn(history, user_message):
    history.append({"role": "user", "content": user_message})

    with weave.start_turn(user_message=user_message, model=MODEL):
        # LLM call 1 — the model may decide to use a tool.
        with weave.start_llm(model=MODEL, provider_name="openai") as llm:
            resp = openai_client.chat.completions.create(
                model=MODEL, messages=history, tools=[wikipedia_tool_schema],
            )
            msg = resp.choices[0].message
            llm.output(msg.content or "")
            llm.usage = weave.Usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
            history.append(msg.model_dump(exclude_none=True))

        # If no tool was requested, the first LLM response is the answer.
        if not msg.tool_calls:
            return msg.content

        # Execute each requested tool call.
        for tc in msg.tool_calls:
            with weave.start_tool(
                name=tc.function.name,
                arguments=tc.function.arguments,
                tool_call_id=tc.id,
            ) as tool:
                tool.result = wikipedia_search(**json.loads(tc.function.arguments))
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool.result,
                })

        # LLM call 2 — synthesize the final answer.
        with weave.start_llm(model=MODEL, provider_name="openai") as llm:
            resp = openai_client.chat.completions.create(model=MODEL, messages=history)
            msg = resp.choices[0].message
            llm.output(msg.content)
            llm.usage = weave.Usage(
                input_tokens=resp.usage.prompt_tokens,
                output_tokens=resp.usage.completion_tokens,
            )
            history.append({"role": "assistant", "content": msg.content})
            return msg.content

with weave.start_session(agent_name="research-bot") as session:
    history = []
    for question in [
        "Who founded Anthropic?",
        "What is Claude (the AI assistant)?",
        "Summarize what we discussed in one sentence.",
    ]:
        print(f"USER: {question}")
        print(f"AGENT: {run_turn(history, question)}\n")
```

## See your agent traces in the Agents view

When `weave.init()` runs, it prints a link to your project where you can see:

- A row in the **Agents** tab for `research-bot`
- One session containing three turns
- Each turn (`invoke_agent`) with two `chat` spans and an `execute_tool` span nested inside
- Token counts, latency, model, and the full message exchange on each `chat`

Click into any turn to inspect the inputs, outputs, tool arguments, and tool results.

## Next steps

* Get a better understanding of how to [trace agents with Weave](weave/guides/tracking/trace-agents) and what features and options are available in the Weave SDK.
* See the [integration section](/weave/guides/integrations) for more options on how to integrate Weave with your agents.