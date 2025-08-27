---
description: How to use W&B with Weave.
menu:
  default:
    identifier: weave
    parent: integrations
title: Weave
weight: 245
---

Use the W&B Weave integration to automatically initialize Weave whenever you
start a W&B run. As long as you call `wandb.init()` and import `weave` in your
process, Weave will auto-initialize and begin capturing traces with zero
additional setup.

[Learn more about Weave here](https://weave-docs.wandb.ai/)!

## Install Weave

Install the Weave library (and W&B if you have not already):

```bash
pip install wandb weave
```

## Auto-initialize Weave with W&B

Initialize a W&B run and import Weave. No extra configuration is required—Weave will auto-init for you.

```python
import wandb
import weave

wandb.init(project="weave-demo")

# Weave is now auto-initialized and ready to capture traces.
# Use your code as usual; traces will be associated with this W&B run.
```

## Start tracking LLM workflows

Weave automatically tracks LLM calls by patching popular LLM libraries like OpenAI, Anthropic, and Gemini. Simply call your LLM as you normally would, and Weave will automatically track the call.

```python
import wandb
import weave
from openai import OpenAI

wandb.init(project="weave-demo")
client = OpenAI()

# Weave will automatically track this call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is the capital of France?"}]
)
```

You can also use Weave to track arbitrary python functions by decorating them with `@weave.op`.

```python
import wandb
import weave

wandb.init(project="weave-demo")

@weave.op
def agent_step(**kwargs):
    ...

def internal_step(**kwargs):
    ...


# Weave will automatically track this call
agent_step()

# Weave will not track this call
internal_step()
```

## View your traces

Open the W&B run page linked in your console output after `wandb.init()`. Your Weave traces will be visible in the run, making it easy to explore, compare, and share results.
