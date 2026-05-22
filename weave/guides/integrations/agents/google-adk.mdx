---
title: "Google ADK"
description: Trace an agent built with Google's Agent Development Kit (ADK) using Weave.
---

[Google's Agent Development Kit (ADK)](https://google.github.io/adk-docs/) is a flexible, model-agnostic Python framework for building and orchestrating agents. While optimized for Gemini, ADK supports any model and supports both simple tasks and complex multi-agent workflows. Weave automatically traces agents built with ADK, including each agent invocation, sub-agent handoff, model call, and tool call. Weave displays the captured data in the **Agents** view of your project.

## Trace Google ADK agents with Weave

The Weave SDK autopatches with Google ADK, allowing you to capture traces from your ADK agents with minimal set up. This doc shows how to initialize Weave and then run a Google ADK Agent (named `weather_assistant`, using `gemini-2.5-flash` with a `get_weather` tool) through an `InMemoryRunner` so that Weave automatically traces the agent invocation, model call, and tool call end-to-end.

### Prerequisites

- A W&B account and [API key](https://wandb.ai/authorize) set as an `WANDB_API_KEY` environment variable
- A [Google API key](https://aistudio.google.com/apikey) for Gemini
- Python 3.10+

### Install packages

Install the following packages in your developer environment:

```bash
pip install weave google-adk
```

### Initialize Weave in your code

Add `weave.init` to the project, along with your W&B team and project names, and then build an agent the way you normally would. The following code creates the agent described in the introduction and runs it while Weave captures its traces.

```python lines
import os
import weave
import asyncio
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types

weave.init("<your-team>/<your-project-name>")

def get_weather(city: str) -> dict:
    """Get the current weather for a given city."""
    return {
        "city": city,
        "temperature_range": "14-20C",
        "conditions": "Sunny with wind.",
    }

agent = Agent(
    name="weather_assistant",
    model="gemini-2.0-flash",
    instruction="You are a helpful weather assistant.",
    tools=[get_weather],
)

async def main():
    runner = InMemoryRunner(agent=agent, app_name="weather-app")
    session = await runner.session_service.create_session(
        app_name="weather-app", user_id="user-1"
    )

    async for event in runner.run_async(
        user_id="user-1",
        session_id=session.id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text="What's the weather in Tokyo?")],
        ),
    ):
        if event.is_final_response() and event.content:
            print(event.content.parts[0].text)

asyncio.run(main())
```

`weave.init()` prints a link to your project when it runs. Click it to inspect the trace and see what the agent did at each step.

Learn how to [navigate the Agents view](link-tbd) in the Weave UI.