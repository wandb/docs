---
title: "Quickstart: Trace an OpenAI Agents SDK agent"
description: Trace an agent built with the OpenAI Agents SDK using Weave.
---

The [OpenAI Agents SDK](https://github.com/openai/openai-agents-python) is a lightweight framework for building agents and multi-agent workflows on top of OpenAI's API. Weave automatically traces agents built with the OpenAI Agents SDK, including each agent invocation, sub-agent handoff, model call, and tool call. Weave displays the captured data in the **Agents** view of your project.

## Trace OpenAI SDK agents with Weave

The Weave SDK autopatches with Google ADK, allowing you to capture traces from your ADK agents with minimal set up. The following example code initializes Weave tracing, defines a `get_weather` function tool and a weather-assistant agent, then runs the agent via the OpenAI Agents SDK Runner so Weave automatically captures the full trace.

### Prerequisites

- A W&B account and [API key](https://wandb.ai/authorize) set as an `WANDB_API_KEY` environment variable
- An [OpenAI API key](https://platform.openai.com/api-keys)
- Python 3.10+

### Install packages

Install the following packages in your developer environment:

```bash
pip install weave openai-agents
```

### Initialize Weave in your code

Add `weave.init` to the project, along with your W&B team and project names, and then build an agent the way you normally would.

```python lines highlight="7-10"
import os
import weave
import asyncio
from pydantic import BaseModel
from agents import Agent, Runner, function_tool

weave.init("<your-team>/<your-project-name>")

class Weather(BaseModel):
    city: str
    temperature_range: str
    conditions: str

@function_tool
def get_weather(city: str) -> Weather:
    return Weather(city=city, temperature_range="14-20C", conditions="Sunny with wind.")

agent = Agent(
    name="Weather assistant",
    instructions="You are a helpful weather assistant.",
    tools=[get_weather],
)

async def main():
    result = await Runner.run(agent, input="What's the weather in Tokyo?")
    print(result.final_output)

asyncio.run(main())
```

`weave.init()` prints a link to your project when it runs. Click it to inspect the trace and see what the agent did at each step.

Learn how to [navigate the Agents view](link-tbd) in the Weave UI.